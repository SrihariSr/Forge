import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import random
from Forge import Tensor
from NeuralNetwork.layers import CharTransformer
from NeuralNetwork.losses import CrossEntropyLoss
from Optim.optimizer import Adam

# Dataset setup
text = "hello world hello world hello world "
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

seq_len = 8
inputs = []
targets = []
for i in range(0, len(text) - seq_len):
    inputs.append([char_to_idx[c] for c in text[i:i + seq_len]])
    targets.append([char_to_idx[c] for c in text[i + 1:i + seq_len + 1]])

embed_dim = 16
ff_dim = 32

def train_and_evaluate(seed, epochs=100):
    # Train a model with a given seed and return the final loss and generated text
    random.seed(seed)

    model = CharTransformer(vocab_size, embed_dim, ff_dim, seq_len)
    criterion = CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=0.001)

    # Train
    final_loss = 0.0
    for epoch in range(epochs):
        total_loss = 0.0
        for i in range(len(inputs)):
            logits = model([inputs[i]])
            loss = criterion(logits, targets[i])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss._data[0]
        final_loss = total_loss / len(inputs)

    # Generate text
    current = [char_to_idx[c] for c in "hello wo"]
    generated = list("hello wo")
    for _ in range(20):
        logits = model([current])
        last_pos_start = (seq_len - 1) * vocab_size
        best_idx = 0
        best_val = logits._data[last_pos_start]
        for j in range(1, vocab_size):
            if logits._data[last_pos_start + j] > best_val:
                best_val = logits._data[last_pos_start + j]
                best_idx = j
        generated.append(idx_to_char[best_idx])
        current = current[1:] + [best_idx]

    return final_loss, "".join(generated)


# --- Search through seeds ---
num_seeds = 20
results = []

print(f"Searching {num_seeds} seeds with 100 epochs each...")
print(f"{'Seed':>6} | {'Loss':>8} | Generated Text")
print("-" * 60)

for seed in range(num_seeds):
    loss, text_out = train_and_evaluate(seed)
    results.append((seed, loss, text_out))
    print(f"{seed:>6} | {loss:>8.4f} | {text_out}")

# --- Find the best ---
results.sort(key=lambda x: x[1])

print("\n" + "=" * 60)
print("TOP 5 SEEDS")
print("=" * 60)
for seed, loss, text_out in results[:5]:
    print(f"Seed {seed:>4} | Loss: {loss:.4f} | {text_out}")

best_seed = results[0][0]
print(f"\nBest seed: {best_seed}")