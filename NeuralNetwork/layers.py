import math
from Forge.tensor import Tensor
from NeuralNetwork.module import Module
from NeuralNetwork.parameter import Parameter
from NeuralNetwork.module import Module
from Forge.CalcLlama.fused_operations import FusedLinearReLU
import random
import array as _array

class Linear(Module):
    """A fully connected layer: output = input @ weight.T + bias"""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()

        k = 1 / math.sqrt(in_features) # Kaiming initialization
        import random

        weight_data = []
        for i in range(out_features):
            row = []
            for j in range(in_features):
                row.append(random.uniform(-k, k))
            weight_data.append(row)

        self.weight = Parameter(Tensor(weight_data))

        if bias:
            bias_data = [random.uniform(-k, k) for _ in range(out_features)]
            self.bias = Parameter(Tensor([bias_data]))
        else:
            self.bias = None

    def forward(self, x):
        output = x @ self.weight.T
        if self.bias is not None:
            output = output + self.bias
        return output

    def __repr__(self):
        in_f = self.weight.shape[1]
        out_f = self.weight.shape[0]
        return f"Linear(in_features={in_f}, out_features={out_f}, bias={self.bias is not None})"

class ReLULayer(Module):
    def __init__(self):
        super().__init__()
    
    def forward(self, x):
        return x.relu()
    
    def __repr__(self):
        return "ReLU()"

class FusedLinearReLULayer(Module):
    """Linear + ReLU fused into one operation for better performance."""

    def __init__(self, in_features, out_features):
        super().__init__()

        k = 1 / math.sqrt(in_features)

        weight_data = []
        for i in range(out_features):
            row = []
            for j in range(in_features):
                row.append(random.uniform(-k, k))
            weight_data.append(row)

        self.weight = Parameter(Tensor(weight_data))

        bias_data = [random.uniform(-k, k) for _ in range(out_features)]
        self.bias = Parameter(Tensor(bias_data))

    def forward(self, x):
        func = FusedLinearReLU()
        result = func.forward(x, self.weight.T, self.bias)

        any_requires_grad = x.requires_grad or self.weight.requires_grad
        if any_requires_grad:
            result.requires_grad = True
            result._grad_fn = func

        return result

    def __repr__(self):
        in_f = self.weight.shape[1]
        out_f = self.weight.shape[0]
        return f"FusedLinearReLU(in_features={in_f}, out_features={out_f})"

class Sequential(Module):
    # A container that runs layers in sequence

    def __init__(self, *layers):
        super().__init__()
        for i, layer in enumerate(layers):
            setattr(self, f'layer_{i}', layer)
        self._layer_list = list(layers)

    def forward(self, x):
        for layer in self._layer_list:
            x = layer(x)
        return x

    def optimize(self):
        # Apply fusion optimizations to this model
        from Forge.CalcLlama.fusion import optimize_model
        self._layer_list = optimize_model(self._layer_list)

        # Re-register modules
        self._modules = {}
        for i, layer in enumerate(self._layer_list):
            setattr(self, f'layer_{i}', layer)

        return self

    def __repr__(self):
        lines = ["Sequential("]
        for i, layer in enumerate(self._layer_list):
            lines.append(f"  ({i}): {layer}")
        lines.append(")")
        return "\n".join(lines)

class Embedding(Module):
    def __init__(self, num_embeddings, embedding_dim):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        weight_data = []
        for i in range(num_embeddings):
            row = [random.uniform(-1.0, 1.0) for _ in range(embedding_dim)]
            weight_data.append(row)

        self.weight = Parameter(Tensor(weight_data))

    def forward(self, indices):
        from Forge.tensor import Tensor
        import array as _array

        batch_size = len(indices)
        seq_len = len(indices[0])
        dim = self.embedding_dim

        new_data = _array.array(self.weight.dtype.typecode, [])

        for b in range(batch_size):
            for s in range(seq_len):
                idx = indices[b][s]
                start = idx * dim
                for d in range(dim):
                    new_data.append(self.weight._data[start + d])

        result = Tensor.__new__(Tensor)
        result._data = new_data
        result.shape = (batch_size, seq_len, dim)
        result.dtype = self.weight.dtype
        result.requires_grad = True
        result.grad = None
        result._grad_fn = None

        weight_ref = self.weight

        def _backward():
            if result.grad is None:
                return

            if weight_ref.grad is None:
                weight_ref.grad = Tensor.zeros(
                    weight_ref.shape[0], weight_ref.shape[1]
                )

            for b in range(batch_size):
                for s in range(seq_len):
                    idx = indices[b][s]
                    for d in range(dim):
                        grad_idx = (b * seq_len + s) * dim + d
                        weight_idx = idx * dim + d
                        weight_ref.grad._data[weight_idx] += result.grad._data[grad_idx]

        result._backward_fn = _backward
        return result

    def __repr__(self):
        return f"Embedding({self.num_embeddings}, {self.embedding_dim})"

class SimpleAttention(Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.embed_dim = embed_dim
        self.scale = embed_dim ** 0.5
        self.query = Linear(embed_dim, embed_dim)
        self.key = Linear(embed_dim, embed_dim)
        self.value = Linear(embed_dim, embed_dim)

    def forward(self, x):
            batch_size = x.shape[0]
            seq_len = x.shape[1]

            results = []

            for b in range(batch_size):
                x_2d = x.select_batch(b)

                q = self.query(x_2d)
                k = self.key(x_2d)
                v = self.value(x_2d)

                scores = (q @ k.T) * (1.0 / self.scale)
                scores = scores.causal_mask()
                weights = scores.softmax()
                output = weights @ v

                results.append(output)

            return results[0] if batch_size == 1 else results

    def __repr__(self):
        return f"SimpleAttention (embed_dim={self.embed_dim})"

class TransformerBlock(Module):
    def __init__(self, embed_dim, ff_dim):
        super().__init__()
        self.attention = SimpleAttention(embed_dim)
        self.ff1 = Linear(embed_dim, ff_dim)
        self.ff2 = Linear(ff_dim, embed_dim)
    
    def forward(self, x):
        # Self-attention
        attended = self.attention(x)

        # Add residual connection
        x_2d = self._to_2d(x)
        hidden = attended + x_2d

        # Feed-forward network
        ff_out = self.ff1(hidden).relu()
        ff_out = self.ff2(ff_out)

        # Add residual connection
        output = ff_out + hidden
        return output
    
    def _to_2d(self, x):
         if len(x.shape) == 2:
             return x
         return x.select_batch(0)

    def __repr__(self):
        return f"TransformerBlock(embed_dim={self.attention.embed_dim})"

class CharTransformer(Module):
    def __init__(self, vocab_size, embed_dim, ff_dim, seq_len):
        super().__init__()
        self.embed_dim = embed_dim
        self.seq_len = seq_len

        self.embedding = Embedding(vocab_size, embed_dim)

        pos_data = [[random.uniform(-1.0, 1.0) for _ in range(embed_dim)] for _ in range(seq_len)]
        self.pos_embedding = Parameter(Tensor(pos_data))

        self.transformer = TransformerBlock(embed_dim, ff_dim)
        self.output_proj = Linear(embed_dim, vocab_size)
    
    def forward(self, indices):
        # Embed characters
        x = self.embedding(indices)
        
        # Inject position embedding
        x += self.pos_embedding

        # Transform
        x = self.transformer(x)

        # Project to vocabulary
        logits = self.output_proj(x)

        return logits
    
    def __repr__(self):
        return (
            f"CharTransformer(\n"
            f"embedding={self.embedding}\n"
            f"transformer={self.transformer}\n"
            f"output_proj={self.output_proj}\n")

class LayerNorm(Module):
    """
    Rescales a row of numbers such that the mean = 0 and standard deviation = 1
    to ensure numerical stability.
    """
    def __init__(self, embed_dim, eps=1e-5):
        super().__init__()
        self.embed_dim = embed_dim
        self.eps = eps
        self.gain = Parameter(Tensor([[1.0] * embed_dim]))
        self.bias = Parameter(Tensor([[0.0] * embed_dim]))
    
    def forward(self, x):
        mean = x.row_mean()
        centred = x - mean
        variance = (centred ** 2).row_mean()
        std = (variance + self.eps).sqrt()
        normalised = centred / std

        return (self.gain * normalised) + self.bias
    
    def __repr__(self):
        return f"LayerNorm(embed_dim={self.embed}, eps={self.eps})"

class MultiHeadAttention(Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** 0.5
        self.out_proj = Linear(embed_dim, embed_dim)

        # Q, K, V values. Assigned one by one: Module only registers a Parameter or
        # a Module, so heads held in a plain list would never reach parameters().
        for n in range(num_heads):
            setattr(self, f"query_{n}", Linear(embed_dim, self.head_dim))
            setattr(self, f"key_{n}", Linear(embed_dim, self.head_dim))
            setattr(self, f"value_{n}", Linear(embed_dim, self.head_dim))

    def query(self, n):
        return getattr(self, f"query_{n}")

    def key(self, n):
        return getattr(self, f"key_{n}")

    def value(self, n):
        return getattr(self, f"value_{n}")

        for n in range(num_heads):
            setattr(self, f"query_{n}", self.queries[n])
            setattr(self, f"key{n}", self.keys[n])
            setattr(self, f"value_{n}", self.values[n])
    def forward(self, x):
        batch_size = x.shape[0]
        results = []

        for b in range(batch_size):
            x_2d = x.select_batch(b)

            head_outputs = []
            for n in range(self.num_heads):
                q = self.query(n)(x_2d)
                k = self.key(n)(x_2d)
                v = self.value(n)(x_2d)

                # How much should each position attend to each other position
                scores = (q @ k.T) / self.scale

                # Masking every position after it
                scores = scores.causal_mask()

                weights = scores.softmax()
                head_outputs.append(weights @ v)

            combined = head_outputs[0].concat_columns(*head_outputs[1:])
            results.append(self.out_proj(combined))

        return results[0] if batch_size == 1 else results
        
    def __repr__(self):
        return (f"MultiHeadAttention(embed_dim={self.embed_dim})\nnum_heads={self.num_heads}\nhead_dim={self.head_dim}")

class GPTBlock(Module):
    def __init__(self, embed_dim, num_heads, ff_dim):
        super().__init__()
        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)
        self.attention = MultiHeadAttention(embed_dim, num_heads)
        self.ff1 = Linear(embed_dim, ff_dim)
        self.ff2 = Linear(ff_dim, embed_dim)

    def forward(self, x_2d):
        # Attention sub-layer
        normed = self.norm1(x_2d)
        attended = self.attention(normed.unsqueeze_batch())
        x_2d += attended # Residual

        # Feed-forward sub-layer
        normed = self.norm2(x_2d)
        ff_out = self.ff2(self.ff1(normed).gelu())
        x_2d += ff_out

        return x_2d

        def __repr__(self):
            return f"GPTBlock(embd_dim={self.attention.embed_dim}), heads={self.attention.num_heads}"
        
class GPT(Module):
    """
    A small decoder-only transformer.
    """
    def __init__(self,
    vocab_size,
    embed_dim,
    num_heads,
    ff_dim,
    num_layers,
    seq_len):
        super().__init__()
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.seq_len = seq_len

        # Token embedding
        self.embedding = Embedding(vocab_size, embed_dim)

        # Learned positional encoding
        pos_data = [[[random.uniform(-0.02, 0.02) for _ in range(embed_dim)] for _ in range(seq_len)]]
        self.pos_embedding = Parameter(Tensor(pos_data))

        # Stacking the GPT blocks together
        self.blocks = [GPTBlock(embed_dim, num_heads, ff_dim) for _ in range(num_layers)]
        for i, block in enumerate(self.blocks):
            setattr(self, f"block_{i}", block)
        # Final normalisation
        self.norm_final = LayerNorm(embed_dim)

        # Project back to one score ("logit") per vocabulary entry
        self.output_proj = Linear(embed_dim, vocab_size)
    
    def forward(self, indices):
        x = self.embedding(indices)
        x = x + self.pos_embedding # Inject position info
        x = x.select_batch(0) # Drop to 2D for the blocks

        for block in self.blocks:
            x = block(x)
        
        x = self.norm_final(x)
        logits = self.output_proj(x)
        return logits

        def __repr__(self):
            return f"GPT(vocab={self.vocab_size}, embed={self.embed_dim}, layers={self.num_layers}, seq_len={self.seq_len})"
        
