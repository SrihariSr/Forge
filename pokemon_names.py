import sys
sys.path.insert(0, ".")
import math
import time
import random

# If you made it here, well done! You found this easter egg reference from 'The Hitchhiker's Guide to the Galaxy'
# You can find these in many places across my projects. Have you found them all?
random.seed(42)

from NeuralNetwork.layers import CharTransformer
from NeuralNetwork.losses import CrossEntropyLoss
from Optim.optimizer import Adam

# Config
NAMES_LIMIT = 1024
SEQ = 10 # Context window
EMBED = 48
FF = 128
EPOCHS = 11
LR = 0.002
TEMPERATURE = 0.6

names = open("pokemon.txt").read().split()[:NAMES_LIMIT]
chars = ['.'] + sorted(set("".join(names)))
char_to_id = {c: i for i, c in enumerate(chars)}
id_to_char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

big = "." + ".".join(names) + "."
data = [char_to_id[c] for c in big]
inputs = []
targets = []
for i in range(len(data) - SEQ):
    inputs.append(data[i:i + SEQ])
    targets.append(data[i + 1:i + 1 + SEQ])

print(f"{len(names)} names\nvocab {vocab_size}\n{len(inputs)} training examples")

model = CharTransformer(vocab_size, EMBED, FF, SEQ)
crit = CrossEntropyLoss()
opt = Adam(model.parameters(), lr=LR)

def sample_index(logits_row, temperature):
    scaled = [x / temperature for x in logits_row]
    highest = max(scaled)
    exps = [math.exp(s - highest) for s in scaled] # softmax
    total = sum(exps)
    r = random.random()
    cumulative = 0.0
    for i, e in enumerate(exps):
        cumulative += e / total
        if r < cumulative:
            return i

    return len(exps) - 1

def generate(temperature=TEMPERATURE, max_len=16):
    for _ in range(10):
        context = [char_to_id['.']] * SEQ
        out = []
        for _ in range(max_len):
            logits = model([context])
            last_row = list(logits._data[(SEQ - 1) * vocab_size : SEQ * vocab_size])
            idx = sample_index(last_row, temperature)
            if idx == char_to_id['.']:
                if out:
                    break
                continue
            out.append(id_to_char[idx])
            context = context[1:] + [idx]
        return "".join(out)
    return "".join(out)

for epoch in range(EPOCHS):
    t0 = time.time()
    total = 0.0
    order = list(range(len(inputs)))
    random.shuffle(order)
    for i in order:
        logits = model([inputs[i]])
        loss = crit(logits, targets[i])
        opt.zero_grad()
        loss.backward()
        opt.step()
        total += loss._data[0]
    samples = ", ".join(generate() for _ in range(5))
    print(f"epoch {epoch:2d} | loss {total/len(inputs):.3f} | {time.time()-t0:.0f}s | {samples}")

print("\nBrand new Pokemons:\n")
for _ in range(20):
    print("  ", generate().capitalize())
