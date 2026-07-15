"""
Trains a GPT on Shakespeare using DATA-PARALLEL training across CPU cores.
"""

import sys
sys.path.insert(0, ".")

import array
import math
import os
import random
import time
import pickle
from multiprocessing import Pool

from Forge import Tensor
from NeuralNetwork.layers import GPT
from NeuralNetwork.losses import CrossEntropyLoss
from Optim.optimizer import Adam

# Config
DATA_FILE = "shakespeare.txt"

NUM_WORKERS = 8 # leave a core free for the OS and the parent process
BATCH_SIZE = 24 # must divide evenly by NUM_WORKERS (26 = 13 x 2)
SEQ_LEN = 32     # context window, in characters

EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256
NUM_LAYERS = 4

LEARNING_RATE = 0.002
STEPS = 25000           # total optimizer steps (not epochs)
SAMPLE_EVERY = 1000     # generate a text sample this often
CHECKPOINT_EVERY = 500 # save weights this often, so a crash costs little

TEMPERATURE = 0.8
SEED = 42

# Worker globals. Each process builds its own model ONCE, then reuses it --
# rebuilding the model every step would waste more time than it saves.
_worker_model = None
_worker_loss = None


def _init_worker(vocab_size, seed):
    """Runs once per worker process, when the Pool starts."""
    global _worker_model, _worker_loss
    random.seed(seed + os.getpid())          # different seed per worker
    _worker_model = GPT(vocab_size, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, SEQ_LEN)
    _worker_loss = CrossEntropyLoss()


def _worker_step(task):
    """
    Runs in a worker process. Receives the current weights and a shard of the
    batch; returns the gradients and the loss for that shard.
    """
    weights, inputs, targets = task

    # Load the parent's current weights into this worker's model.
    # _data is an array.array, so a slice can only take another array.array.
    for param, values in zip(_worker_model.parameters(), weights):
        param._data[:] = array.array(param._data.typecode, values)

    # Forward + backward on this shard only.
    logits = _worker_model(inputs)
    loss = _worker_loss(logits, targets)
    _worker_model.zero_grad()
    loss.backward()

    # Ship the gradients back as plain lists (they pickle cheaply).
    grads = []
    for param in _worker_model.parameters():
        if param.grad is None:
            grads.append(None)
        else:
            grads.append(list(param.grad._data))

    return grads, loss._data[0]

# Text generation (runs in the parent, single process)
def sample_index(logits_row, temperature):
    """Pick a character index by weighted random choice, sharpened by temperature."""
    scaled = [x / temperature for x in logits_row]
    highest = max(scaled)
    exps = [math.exp(s - highest) for s in scaled]
    total = sum(exps)

    r = random.random()
    cumulative = 0.0
    for i, e in enumerate(exps):
        cumulative += e / total
        if r < cumulative:
            return i
    return len(exps) - 1


def generate(model, char_to_id, id_to_char, vocab_size, prompt="\n", length=300):
    """Generate text one character at a time, feeding each prediction back in."""
    context = [char_to_id.get(c, 0) for c in prompt][-SEQ_LEN:]
    # left-pad so the context is always exactly SEQ_LEN long
    while len(context) < SEQ_LEN:
        context = [char_to_id.get("\n", 0)] + context

    out = []
    for _ in range(length):
        logits = model([context])
        # logits is (SEQ_LEN, vocab); we want the LAST position's prediction
        last_row = list(logits._data[(SEQ_LEN - 1) * vocab_size: SEQ_LEN * vocab_size])
        idx = sample_index(last_row, TEMPERATURE)
        out.append(id_to_char[idx])
        context = context[1:] + [idx]

    return "".join(out)

def main():
    random.seed(SEED)

    # load and encode the text
    text = open(DATA_FILE, encoding="utf-8").read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for i, c in enumerate(chars)}
    data = [char_to_id[c] for c in text]

    print(f"corpus     : {len(text):,} characters, vocab {vocab_size}")
    print(f"model      : {NUM_LAYERS} layers, {NUM_HEADS} heads, {EMBED_DIM} dim")
    print(f"parallelism: {NUM_WORKERS} workers x {BATCH_SIZE // NUM_WORKERS} sequences "
          f"= batch {BATCH_SIZE}")
    print()

    if BATCH_SIZE % NUM_WORKERS != 0:
        raise ValueError("BATCH_SIZE must divide evenly by NUM_WORKERS")

    per_worker = BATCH_SIZE // NUM_WORKERS

    # the master model and optimizer live HERE, in the parent
    model = GPT(vocab_size, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, SEQ_LEN)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)
    params = list(model.parameters())
    print(f"parameters : {len(params)} tensors\n")

    def random_batch(n):
        """Grab n random windows of text. Each target is the input shifted by one."""
        inputs, targets = [], []
        for _ in range(n):
            i = random.randrange(0, len(data) - SEQ_LEN - 1)
            inputs.append(data[i:i + SEQ_LEN])
            targets.extend(data[i + 1:i + SEQ_LEN + 1])
        return inputs, targets
    
    pool = Pool(processes=NUM_WORKERS,
                initializer=_init_worker,
                initargs=(vocab_size, SEED),
                maxtasksperchild=10)


    print("training...\n")
    start = time.time()
    running_loss = None

    try:
        for step in range(1, STEPS + 1):
            # Snapshot the current weights to hand out to the workers.
            weights = [list(p._data) for p in params]

            # Build one shard per worker.
            tasks = []
            for _ in range(NUM_WORKERS):
                inputs, targets = random_batch(per_worker)
                tasks.append((weights, inputs, targets))

            # Run all workers in parallel and collect their results.
            results = pool.map(_worker_step, tasks)

            # average the gradients across workers
            optimizer.zero_grad()
            for i, param in enumerate(params):
                total = None
                for grads, _ in results:
                    g = grads[i]
                    if g is None:
                        continue
                    if total is None:
                        total = list(g)
                    else:
                        for j in range(len(total)):
                            total[j] += g[j]

                if total is None:
                    continue

                for j in range(len(total)):
                    total[j] /= NUM_WORKERS

                if param.grad is None:
                    param.grad = Tensor.zeros(*param.shape)
                param.grad._data[:] = array.array(param.grad._data.typecode, total)

            # One optimizer step, in the parent, using the averaged gradients.
            optimizer.step()

            # reporting
            loss = sum(l for _, l in results) / len(results)
            running_loss = loss if running_loss is None else 0.9 * running_loss + 0.1 * loss

            if step % 10 == 0:
                elapsed = time.time() - start
                rate = step / elapsed
                eta = (STEPS - step) / rate / 60
                print(f"step {step:5d}/{STEPS} | loss {running_loss:6.3f} | "
                      f"{rate:5.2f} steps/s | ETA {eta:5.1f} min", flush=True)

            if step % SAMPLE_EVERY == 0:
                print("\n--- sample ---")
                print(generate(model, char_to_id, id_to_char, vocab_size, length=280))
                print("--- end ---\n", flush=True)

            if step % CHECKPOINT_EVERY == 0:
                with open("shakespeare_checkpoint.pkl", "wb") as f:
                    pickle.dump({
                        "weights": [list(p._data) for p in params],
                        "chars": chars,
                        "step": step,
                    }, f)

    except KeyboardInterrupt:
        print("\ninterrupted -- saving checkpoint")

    finally:
        pool.close()
        pool.join()

    with open("shakespeare_checkpoint.pkl", "wb") as f:
        pickle.dump({
            "weights": [list(p._data) for p in params],
            "chars": chars,
            "step": STEPS,
        }, f)

    print("\n" + "=" * 60)
    print("FINAL SAMPLE")
    print("=" * 60)
    print(generate(model, char_to_id, id_to_char, vocab_size, length=1000))


if __name__ == "__main__":
    main()