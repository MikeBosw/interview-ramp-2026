"""
The most atomic way to train and run inference for a GPT in pure, dependency-free Python.
This file is the complete algorithm.
Everything else is just efficiency.

@karpathy
"""

import math  # math.log, math.exp
import os  # os.path.exists
import random  # random.seed, random.choices, random.gauss, random.shuffle
import sys
import urllib.request
from dataclasses import dataclass
from typing import Callable, Iterable, Mapping, Sequence

type Matrix[T] = list[list[T]]
type ImmMatrix[T] = Sequence[Sequence[T]]


@dataclass(frozen=True)
class Emission:
    value: str
    end: str = "\n"


# Let there be Autograd to recursively apply the chain rule through a computation graph
class Value:
    __slots__ = ("data", "grad", "_children", "_local_grads")  # Python optimization for memory usage

    def __init__(self, data: float, children: Iterable[Value] = (), local_grads: Iterable[float] = ()):
        self.data = data  # scalar value of this node calculated during forward pass
        self.grad: float = 0.0  # derivative of the loss w.r.t. this node, calculated in backward pass
        self._children = children  # children of this node in the computation graph
        self._local_grads = local_grads  # local derivative of this node w.r.t. its children

    @classmethod
    def of(cls, other: Value | float | int) -> Value:
        if isinstance(other, Value):
            return other
        if isinstance(other, int):
            return Value(float(other))
        if isinstance(other, float):
            return Value(other)
        raise ValueError(f"Cannot convert {type(other)} to Value: {other}")

    def __add__(self, other_value: Value | float) -> Value:
        other = other_value if isinstance(other_value, Value) else Value(other_value)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other_value: Value | float) -> Value:
        other = other_value if isinstance(other_value, Value) else Value(other_value)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other: Value | float) -> Value:
        other_data = other.data if isinstance(other, Value) else other
        local_grad: float = other_data * self.data ** (other_data - 1)
        return Value(self.data**other_data, (self,), (local_grad,))

    def log(self) -> Value:
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self) -> Value:
        return Value(math.exp(self.data), (self,), (math.exp(self.data),))

    def relu(self) -> Value:
        return Value(max(0, self.data), (self,), (float(self.data > 0),))

    def __neg__(self) -> Value:
        return self * -1

    def __radd__(self, other: Value | float) -> Value:
        return self + other

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other)

    def __rsub__(self, other: Value | float) -> Value:
        return other + (-self)

    def __rmul__(self, other: Value | float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        return self * other**-1

    def __rtruediv__(self, other: Value | float) -> Value:
        return other * self**-1

    def backward(self) -> None:
        topo: list[Value] = []
        visited: set[Value] = set()

        def build_topo(v: Value) -> None:
            if v not in visited:
                visited.add(v)
                for child in v._children:
                    build_topo(child)
                topo.append(v)

        build_topo(self)
        self.grad = 1
        for v in reversed(topo):
            for child, local_grad in zip(v._children, v._local_grads):
                child.grad += local_grad * v.grad


State = Mapping[str, ImmMatrix[Value]]


# Define the model architecture: a function mapping tokens and parameters to logits over what comes next
# Follow GPT-2, blessed among the GPTs, with minor differences: layernorm -> rmsnorm, no biases, GeLU -> ReLU
def linear(x: list[Value], w: ImmMatrix[Value]) -> list[Value]:
    return [Value.of(sum(wi * xi for wi, xi in zip(wo, x))) for wo in w]


def softmax(logits: list[Value]) -> list[Value]:
    max_val = max(val.data for val in logits)
    exps = [(val - max_val).exp() for val in logits]
    total = sum(exps)
    return [e / total for e in exps]


def rmsnorm(x: list[Value]) -> list[Value]:
    ms = sum(xi * xi for xi in x) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


class Gpt:
    def __init__(self, n_layer: int) -> None:
        self.n_layer = n_layer
        self._keys_mut: Sequence[Matrix[Value]] = [[] for _ in range(n_layer)]
        self._values_mut: Sequence[Matrix[Value]] = [[] for _ in range(n_layer)]

    def train(self, token_id: int, pos_id: int, state_dict: State, n_head: int, head_dim: int) -> list[Value]:
        keys, values = self._keys_mut, self._values_mut
        tok_emb = state_dict["wte"][token_id]  # token embedding
        pos_emb = state_dict["wpe"][pos_id]  # position embedding
        x = [t + p for t, p in zip(tok_emb, pos_emb)]  # joint token and position embedding
        x = rmsnorm(x)  # note: not redundant due to backward pass via the residual connection

        for li in range(self.n_layer):
            # 1) Multi-head Attention block
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, state_dict[f"layer{li}.attn_wq"])
            k = linear(x, state_dict[f"layer{li}.attn_wk"])
            v = linear(x, state_dict[f"layer{li}.attn_wv"])
            key_for_layer = keys[li]
            key_for_layer.append(k)
            values[li].append(v)
            x_attn = []
            for h in range(n_head):
                hs = h * head_dim
                q_h = q[hs : hs + head_dim]
                k_h = [ki[hs : hs + head_dim] for ki in key_for_layer]
                v_h = [vi[hs : hs + head_dim] for vi in values[li]]
                attn_logits = [
                    Value.of(sum(q_h[j] * k_h[t][j] for j in range(head_dim)) / head_dim**0.5) for t in range(len(k_h))
                ]
                attn_weights = softmax(attn_logits)
                head_out = [
                    Value.of(sum(attn_weights[t] * v_h[t][j] for t in range(len(v_h)))) for j in range(head_dim)
                ]
                x_attn.extend(head_out)
            x = linear(x_attn, state_dict[f"layer{li}.attn_wo"])
            x = [a + b for a, b in zip(x, x_residual)]
            # 2) MLP block
            x_residual = x
            x = rmsnorm(x)
            x = linear(x, state_dict[f"layer{li}.mlp_fc1"])
            x = [xi.relu() for xi in x]
            x = linear(x, state_dict[f"layer{li}.mlp_fc2"])
            x = [a + b for a, b in zip(x, x_residual)]

        logits = linear(x, state_dict["lm_head"])
        return logits


def main(num_training_steps: int = 1000, emit: Callable[[str], None] = lambda emission: print(emission)) -> None:
    random.seed(42)  # Let there be order among chaos

    # Let there be a Dataset `docs`: list[str] of documents (e.g. a list of names)
    if not os.path.exists("./var/input.txt"):
        names_url = "https://raw.githubusercontent.com/karpathy/makemore/988aa59/names.txt"
        urllib.request.urlretrieve(names_url, "./var/input.txt")
    docs = [line.strip() for line in open("./var/input.txt") if line.strip()]
    random.shuffle(docs)
    emit(f"num docs: {len(docs)}")

    # Let there be a Tokenizer to translate strings to sequences of integers ("tokens") and back
    uchars = sorted(set("".join(docs)))  # unique characters in the dataset become token ids 0..n-1
    BOS = len(uchars)  # token id for a special Beginning of Sequence (BOS) token
    vocab_size = len(uchars) + 1  # total number of unique tokens, +1 is for BOS
    emit(f"vocab size: {vocab_size}")

    # Initialize the parameters, to store the knowledge of the model
    n_layer = 1  # depth of the transformer neural network (number of layers)
    n_embd = 16  # width of the network (embedding dimension)
    block_size = 16  # maximum context length of the attention window (note: the longest name is 15 characters)
    n_head = 4  # number of attention heads
    head_dim = n_embd // n_head  # derived dimension of each head
    state_dict: State = blank_state(block_size, n_embd, n_layer, vocab_size)
    params: Sequence[Value] = [
        p for mat in state_dict.values() for row in mat for p in row
    ]  # flatten params into a single list[Value]
    emit(f"num params: {len(params)}")

    # Let there be Adam, the blessed optimizer and its buffers
    learning_rate, beta1, beta2, eps_adam = 0.01, 0.85, 0.99, 1e-8
    m = [0.0] * len(params)  # first moment buffer
    v = [0.0] * len(params)  # second moment buffer

    # Repeat in sequence
    for step in range(num_training_steps):
        # Take single document, tokenize it, surround it with BOS special token on both sides
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(ch) for ch in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        # Forward the token sequence through the model, building up the computation graph all the way to the loss
        gpt = Gpt(n_layer)
        losses = []
        for pos_id in range(n):
            token_id, target_id = tokens[pos_id], tokens[pos_id + 1]
            logits = gpt.train(token_id, pos_id, state_dict, n_head, head_dim)
            probs = softmax(logits)
            loss_t = -probs[target_id].log()
            losses.append(loss_t)
        loss = Value.of((1 / n) * sum(losses))  # final average loss over the document sequence. May yours be low.

        # Backward the loss, calculating the gradients with respect to all model parameters
        loss.backward()

        # Adam optimizer update: update the model parameters based on the corresponding gradients
        lr_t = learning_rate * (1 - step / num_training_steps)  # linear learning rate decay
        for i, p in enumerate(params):
            m[i] = beta1 * m[i] + (1 - beta1) * p.grad
            v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
            m_hat = m[i] / (1 - beta1 ** (step + 1))
            v_hat = v[i] / (1 - beta2 ** (step + 1))
            p.data -= lr_t * m_hat / (v_hat**0.5 + eps_adam)
            p.grad = 0

        print(f"step {step + 1:4d} / {num_training_steps:4d} | loss {loss.data:.4f}", end="\r", file=sys.stderr)

    # Inference: may the model babble back to us
    temperature = 0.5  # in (0, 1], control the "creativity" of generated text, low to high
    emit("\n--- inference (new, hallucinated names) ---")
    for sample_idx in range(20):
        gpt = Gpt(n_layer)
        token_id = BOS
        sample = []
        for pos_id in range(block_size):
            logits = gpt.train(token_id, pos_id, state_dict, n_head, head_dim)
            probs = softmax([logit / temperature for logit in logits])
            token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
            if token_id == BOS:
                break
            sample.append(uchars[token_id])
        emit(f"sample {sample_idx + 1:2d}: {''.join(sample)}")


def blank_state(block_size: int, n_embd: int, n_layer: int, vocab_size: int) -> State:
    state: dict[str, Matrix[Value]] = {
        "wte": matrix(vocab_size, n_embd),
        "wpe": matrix(block_size, n_embd),
        "lm_head": matrix(vocab_size, n_embd),
    }
    for i in range(n_layer):
        state[f"layer{i}.attn_wq"] = matrix(n_embd, n_embd)
        state[f"layer{i}.attn_wk"] = matrix(n_embd, n_embd)
        state[f"layer{i}.attn_wv"] = matrix(n_embd, n_embd)
        state[f"layer{i}.attn_wo"] = matrix(n_embd, n_embd)
        state[f"layer{i}.mlp_fc1"] = matrix(4 * n_embd, n_embd)
        state[f"layer{i}.mlp_fc2"] = matrix(n_embd, 4 * n_embd)
    return state


def matrix(nout: int, nin: int) -> Matrix[Value]:
    return [[Value(random.gauss(0, 0.08)) for _ in range(nin)] for _ in range(nout)]


if __name__ == "__main__":
    main()
