import sys
sys.path.insert(0, ".")

import math
import os
import random
import time
import pickle
from multiprocessing import Pool
import array as _array
from Forge import Tensor
from NeuralNetwork.layers import GPT
from NeuralNetwork.losses import CrossEntropyLoss
from Optim.optimizer import Adam

DATA_FILE = "shakespeare.txt"
CHECKPOINT_IN = "shakespeare_checkpoint.pkl"
CHECKPOINT_OUT = "shakespeare_checkpoint.pkl"   # overwrite the same file

TRAIN_HOURS = 2.0        # stop after this many hours

NUM_WORKERS = 8          # keep the same as your successful run
BATCH_SIZE = 24          # 8 x 3
SEQ_LEN = 32

# These FOUR must exactly match the run that produced the checkpoint:
EMBED_DIM = 128
NUM_HEADS = 4
FF_DIM = 256
NUM_LAYERS = 4

LEARNING_RATE = 0.002
SAMPLE_EVERY = 1000
CHECKPOINT_EVERY = 500

TEMPERATURE = 0.8
SEED = 43                # different seed than the first run


_worker_model = None
_worker_loss = None


def _init_worker(vocab_size, seed):
    global _worker_model, _worker_loss
    random.seed(seed + os.getpid())
    _worker_model = GPT(vocab_size, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, SEQ_LEN)
    _worker_loss = CrossEntropyLoss()


def _worker_step(task):
    weights, inputs, targets = task
    for param, values in zip(_worker_model.parameters(), weights):
        param._data[:] = _array.array(param._data.typecode, values)
    logits = _worker_model(inputs)
    loss = _worker_loss(logits, targets)
    _worker_model.zero_grad()
    loss.backward()
    grads = []
    for param in _worker_model.parameters():
        grads.append(None if param.grad is None else list(param.grad._data))
    return grads, loss._data[0]


def sample_index(logits_row, temperature):
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


def generate(model, char_to_id, id_to_char, vocab_size, prompt="\n", length=400):
    context = [char_to_id.get(c, 0) for c in prompt][-SEQ_LEN:]
    while len(context) < SEQ_LEN:
        context = [char_to_id.get("\n", 0)] + context
    out = []
    for _ in range(length):
        logits = model([context])
        last_row = list(logits._data[(SEQ_LEN - 1) * vocab_size: SEQ_LEN * vocab_size])
        idx = sample_index(last_row, TEMPERATURE)
        out.append(id_to_char[idx])
        context = context[1:] + [idx]
    return "".join(out)


def _save(params, optimizer, chars, step):
    """Save weights AND Adam's per-parameter state, so a later resume is clean."""
    state = {
        "weights": [list(p._data) for p in params],
        "chars": chars,
        "step": step,
    }
    # Pull out Adam's internal moment buffers if they exist (attribute names may
    # vary; we grab whatever list-like per-parameter state the optimizer holds).
    opt_state = {}
    for attr in ("m", "v", "t", "step_count", "moment1", "moment2"):
        if hasattr(optimizer, attr):
            val = getattr(optimizer, attr)
            try:
                opt_state[attr] = pickle.loads(pickle.dumps(val))
            except Exception:
                pass
    state["optimizer"] = opt_state
    with open(CHECKPOINT_OUT, "wb") as f:
        pickle.dump(state, f)


def main():
    random.seed(SEED)

    # --- load text ---
    text = open(DATA_FILE, encoding="utf-8").read()
    chars = sorted(set(text))
    vocab_size = len(chars)
    char_to_id = {c: i for i, c in enumerate(chars)}
    id_to_char = {i: c for i, c in enumerate(chars)}
    data = [char_to_id[c] for c in text]

    # --- load checkpoint ---
    with open(CHECKPOINT_IN, "rb") as f:
        ckpt = pickle.load(f)
    start_step = ckpt.get("step", 0)
    print(f"resuming from checkpoint at step {start_step}")
    print(f"corpus: {len(text):,} chars, vocab {vocab_size}")
    print(f"will train for {TRAIN_HOURS} hours with {NUM_WORKERS} workers\n")

    # --- build model, load weights ---
    model = GPT(vocab_size, EMBED_DIM, NUM_HEADS, FF_DIM, NUM_LAYERS, SEQ_LEN)
    params = list(model.parameters())
    if len(params) != len(ckpt["weights"]):
        raise ValueError(
            f"checkpoint has {len(ckpt['weights'])} tensors but this model has "
            f"{len(params)} -- your EMBED_DIM/NUM_LAYERS/etc must match the run "
            f"that made the checkpoint."
        )
    for p, values in zip(params, ckpt["weights"]):
        p._data[:] = _array.array(p._data.typecode, values)

    optimizer = Adam(params, lr=LEARNING_RATE)
    # restore Adam state if the checkpoint has it
    for attr, val in ckpt.get("optimizer", {}).items():
        if hasattr(optimizer, attr):
            setattr(optimizer, attr, val)

    def random_batch(n):
        inputs, targets = [], []
        for _ in range(n):
            i = random.randrange(0, len(data) - SEQ_LEN - 1)
            inputs.append(data[i:i + SEQ_LEN])
            targets.extend(data[i + 1:i + SEQ_LEN + 1])
        return inputs, targets

    per_worker = BATCH_SIZE // NUM_WORKERS
    pool = Pool(processes=NUM_WORKERS, initializer=_init_worker,
                initargs=(vocab_size, SEED), maxtasksperchild=10)

    deadline = time.time() + TRAIN_HOURS * 3600
    start = time.time()
    step = start_step
    running_loss = None

    print("training...\n", flush=True)
    try:
        while time.time() < deadline:
            step += 1
            weights = [list(p._data) for p in params]
            tasks = [(weights, *random_batch(per_worker)) for _ in range(NUM_WORKERS)]
            results = pool.map(_worker_step, tasks)

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
                param.grad._data[:] = _array.array(param.grad._data.typecode, total)
            optimizer.step()

            loss = sum(l for _, l in results) / len(results)
            running_loss = loss if running_loss is None else 0.9 * running_loss + 0.1 * loss

            if step % 10 == 0:
                mins_left = (deadline - time.time()) / 60
                print(f"step {step:6d} | loss {running_loss:6.3f} | "
                      f"{mins_left:5.1f} min left", flush=True)

            if step % SAMPLE_EVERY == 0:
                print("\n--- sample ---")
                print(generate(model, char_to_id, id_to_char, vocab_size, length=300))
                print("--- end ---\n", flush=True)

            if step % CHECKPOINT_EVERY == 0:
                _save(params, optimizer, chars, step)

    except KeyboardInterrupt:
        print("\ninterrupted -- saving")
    finally:
        pool.close()
        pool.join()

    _save(params, optimizer, chars, step)
    print(f"\ndone at step {step}. trained for {(time.time()-start)/60:.0f} min.")
    print("\n" + "=" * 60)
    print("FINAL SAMPLE")
    print("=" * 60)
    print(generate(model, char_to_id, id_to_char, vocab_size, length=1200))


if __name__ == "__main__":
    main()