import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import random
from Forge import Tensor
from NeuralNetwork.layers import CharTransformer
from NeuralNetwork.losses import CrossEntropyLoss
from Optim.optimizer import Adam

random.seed(4)

# The text for the model to learn
text = "hello world hello world hello world "

# Build vocabulary
chars = sorted(set(text))
char_to_idx = {c: i for i, c in enumerate(chars)}
idx_to_char = {i: c for i, c in enumerate(chars)}
vocab_size = len(chars)

print(f"Vocabulary: {chars}")
print(f"Vocab size: {vocab_size}")
print(f"Character mapping: {char_to_idx}")

# Create training sequences
seq_len = 8

inputs = []
targets = []

for i in range(0, len(text) - seq_len):
    input_seq = [char_to_idx[c] for c in text[i:i + seq_len]]
    target_seq = [char_to_idx[c] for c in text[i + 1:i + seq_len + 1]]
    inputs.append(input_seq)
    targets.append(target_seq)

print(f"\nNumber of training sequences: {len(inputs)}")
print(f"First input:  {inputs[0]} = {''.join(idx_to_char[i] for i in inputs[0])}")
print(f"First target: {targets[0]} = {''.join(idx_to_char[i] for i in targets[0])}")

# Creating the model
embed_dim = 16
ff_dim = 32

model = CharTransformer(vocab_size, embed_dim, ff_dim, seq_len)
criterion = CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001)

print(f"\nModel:\n{model}")
print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")

# Training
print("\nTraining...")
for epoch in range(100):
    total_loss = 0.0
    num_batches = 0

    for i in range(len(inputs)):
        # Forward pass
        logits = model([inputs[i]])

        # Compute loss
        loss = criterion(logits, targets[i])

        # Backward pass
        optimizer.zero_grad()
        loss.backward()

        # Update weights
        optimizer.step()

        total_loss += loss._data[0]
        num_batches += 1

    avg_loss = total_loss / num_batches
    if epoch % 10 == 0:
        print(f"Epoch {epoch:3d}, Loss: {avg_loss:.4f}")

# Generating text
print("\nGenerating text...")
seed = "hello wo"
generated = list(seed)
current = [char_to_idx[c] for c in seed]

for _ in range(20):
    # Run model on current sequence
    logits = model([current])

    # Get scores for last position only
    last_pos_start = (seq_len - 1) * vocab_size
    last_logits = []
    for j in range(vocab_size):
        last_logits.append(logits._data[last_pos_start + j])

    # Picking the character with the highest score
    best_idx = 0
    best_val = last_logits[0]
    for j in range(1, vocab_size):
        if last_logits[j] > best_val:
            best_val = last_logits[j]
            best_idx = j

    next_char = idx_to_char[best_idx]
    generated.append(next_char)

    # Slide window forward
    current = current[1:] + [best_idx]

print("Generated: " + "".join(generated))

from Forge.serialization import save_model, load_model
from NeuralNetwork.layers import CharTransformer

# Save the trained model
save_model(model, "transformer_weights.json")

# Create a fresh model with random weights
model2 = CharTransformer(vocab_size, embed_dim, ff_dim, seq_len)

# Generate with the untrained model (should be garbage)
current = [char_to_idx[c] for c in "hello wo"]
generated_before = list("hello wo")
for _ in range(20):
    logits = model2([current])
    last_pos_start = (seq_len - 1) * vocab_size
    best_idx = 0
    best_val = logits._data[last_pos_start]
    for j in range(1, vocab_size):
        if logits._data[last_pos_start + j] > best_val:
            best_val = logits._data[last_pos_start + j]
            best_idx = j
    generated_before.append(idx_to_char[best_idx])
    current = current[1:] + [best_idx]
print(f"Before loading: {''.join(generated_before)}")

# Load trained weights
load_model(model2, "transformer_weights.json")

# Generate with loaded model (should match original)
current = [char_to_idx[c] for c in "hello wo"]
generated_after = list("hello wo")
for _ in range(20):
    logits = model2([current])
    last_pos_start = (seq_len - 1) * vocab_size
    best_idx = 0
    best_val = logits._data[last_pos_start]
    for j in range(1, vocab_size):
        if logits._data[last_pos_start + j] > best_val:
            best_val = logits._data[last_pos_start + j]
            best_idx = j
    generated_after.append(idx_to_char[best_idx])
    current = current[1:] + [best_idx]
print(f"After loading:  {''.join(generated_after)}")