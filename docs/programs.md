# Programs: several recorded contractions per network

**In short.** A loss may contract a network in several ways (here: the
amplitudes and the norm). Set `net.max_programs = None` once and call the
network as usual, `net(data, *args)`, for each contraction. The first call
with each set of `contract` arguments records it; every later call with the
same arguments replays it. Nothing has to be declared before the loop.
`trace` is optional: it frees intermediate tensors as soon as they are used
(memory, not speed).

This document has four parts: the recipe for a training loop (§1), what
happens underneath (§2), design choices (§3) and implementation (§4). It
builds on `docs/custom_operations.md` (operations and successors).


## 1. A training loop with several contractions

### 1.1 Example: negative log-likelihood of a Born machine

$-\ln p(x) = -\ln|\psi(x)|^2 + \ln Z$ needs two contractions per step,
both before the first `backward`: the amplitudes $\psi(x)$ of the batch,
and $Z = \langle\psi|\psi
angle$. A network whose `contract` takes an
argument that says which:

```python
import torch
import tensorkrowch as tk


class BornChain(tk.TensorNetwork):
    """A real MPS psi(x) = A_0[x_0] ... A_{n-1}[x_{n-1}], open boundaries."""

    def __init__(self, n, d, r):
        super().__init__(name='born_chain')
        dims = [1] + [r] * (n - 1) + [1]
        self.cores = [tk.randn(shape=(dims[i], d, dims[i + 1]),
                               axes_names=('left', 'input', 'right'),
                               name='core', network=self, param_node=True,
                               std=0.5, dtype=torch.float64)
                      for i in range(n)]
        for a, b in zip(self.cores[:-1], self.cores[1:]):
            a['right'] ^ b['left']

    def set_data_nodes(self):
        super().set_data_nodes([c['input'] for c in self.cores],
                               num_batch_edges=1)

    def contract(self, norm=False):
        if not norm:
            # psi(x), zipped from the left: a (batch, r) environment
            env = self.cores[0] @ self.cores[0].neighbours('input')
            for c in self.cores[1:]:
                env = (env @ c) @ c.neighbours('input')
            return env
        # Z = <psi|psi>: a ket and a bra copy of each core, sharing its
        # memory (one edge per axis: the cores' input edges are taken by
        # their data nodes)
        ket, bra = [], []
        for c in self.cores:
            for layer in (ket, bra):
                copy = tk.ParamNode(shape=c.shape, axes_names=c.axes_names,
                                    name='virtual_result_copy',
                                    network=self, virtual=True)
                copy.set_tensor_from(c)
                layer.append(copy)
        for layer in (ket, bra):
            for a, b in zip(layer[:-1], layer[1:]):
                a['right'] ^ b['left']
        for k, b in zip(ket, bra):
            k['input'] ^ b['input']
        env = ket[0] @ bra[0]                    # zipped: an (r, r) environment
        for k, b in zip(ket[1:], bra[1:]):
            env = (env @ k) @ b
        return env


def nll(net, x):
    log_psi2 = 2 * net(x).abs().log()     # program of contract()
    log_Z = net(norm=True).log()          # program of contract(norm=True)
    return -(log_psi2 - log_Z).mean()


n, d, r = 12, 3, 8
net = BornChain(n, d, r)
net.max_programs = None        # keep both programs (default: 1)
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

x = torch.nn.functional.one_hot(torch.randint(d, (64, n)), d).double()
for step in range(100):
    optimizer.zero_grad()
    loss = nll(net, x)         # step 0 records both programs, then replays
    loss.backward()
    optimizer.step()
print(f'{loss.item():.3f}', len(net._programs))
```

Output: `9.224 2` (the loss after 100 steps; two programs). The loss and
its gradients equal those of a dense contraction.

What happens in the loop:

| call | first time | every later time |
|---|---|---|
| `net(x)` | sets the data nodes, adds `x`, runs `contract()` and records its operations: program 1 | adds `x`, replays program 1 |
| `net(norm=True)` | runs `contract(norm=True)` (makes the copies) and records: program 2 | replays program 2 |

"Replays" means: each recorded operation computes its tensor into the node
it created the first time. `contract`'s Python (the loops, the copies, the
wiring) does not run again.

### 1.2 What you have to do

1. **Write `contract` with the contraction choices as arguments** (here
   `norm`). Only arguments that change *which* contraction is done belong
   there; the data goes through `net(data, ...)`. Arguments are compared
   after normalisation (`net(norm=True)` and `net(None, True)` are the same
   program); they must be hashable (lists, tuples, dicts and sets of
   hashable objects are turned into tuples).
2. **Set `max_programs`** to at least the number of different argument sets
   the loss uses (`None`: no limit). With the default, 1, every switch
   between `net(x)` and `net(norm=True)` warns, resets the network and
   records again: 9.1 ms per training step above, against 3.3 ms.
3. **Create the optimizer after the first step** if the network stacks its
   parameters (`auto_stack`, as tk's models do): the first recording
   replaces the parameters of stacked nodes by the stack's. (The example
   does not stack, so it creates the optimizer first.)
4. **Optionally `trace`** (§1.3).
5. **Choose the contraction order well**: the order `contract` uses is what
   is replayed. The zip-up order above keeps (r, r) environments; contracting
   all transfer matrices first would build `n` tensors of r⁴ entries.

### 1.3 `trace`: memory, not speed

`forward` replays from the second call whether or not the network is
traced. Tracing a program makes it erase each intermediate tensor (the
resultant nodes' and the data nodes' tensors) as soon as the program has
read it for the last time, so the network holds nothing between calls:

| n = 100, r = 64, batch 256, under `torch.no_grad` | one call | intermediates the network holds afterwards |
|---|---|---|
| not traced | 27 ms | 64 MB |
| traced | 27 ms | 0 MB |

With autograd, the graph keeps what `backward` needs anyway; tracing then
only drops the network's own references. Without autograd (evaluation,
sampling), it also lowers the peak during the call.

Today a program is traced by calling `trace` with its arguments, once per
program, before or after the first step:

```python
net.max_programs = None
net.trace(x[:1])               # program of contract()
net.trace(None, norm=True)     # program of contract(norm=True)
```

Every recording already counts how often each tensor is read; `trace` only
switches the freeing on (and records the program first if it was not). If
the network keeps `max_programs` programs already, `trace` resets it
first. (Whether `trace` should instead be one switch for all programs is an
open question, §3.8.)

### 1.4 Inputs

A program reads the tensors its nodes hold when it is called, as a recorded
CUDA graph reads its input buffers:

- the network's `data` nodes, set by `net(data, ...)`;
- the parameters, read through any node that shares their memory (the
  copies made with `set_tensor_from` above);
- any other node whose tensor is set before the call (`node.tensor = t`).

### 1.5 What does not work yet: a different set of observed variables per contraction

A loss like the chain rule over an ordering,
$-\ln p(x) = -\sum_k \ln p(x_k \mid x_{<k})$, with
$p(x_k \mid x_{<k}) = P(x_{\le k}) / P(x_{<k})$, needs one marginal per
prefix: a contraction where $x_{<k}$ are observed (contracted with
$arphi(x_i)$) and the rest summed. Each pattern needs its own input
nodes (the feature leg of a variable is connected to its data in one pattern
and summed in another), and the network's data nodes are one set, attached
once. Today the subclass would create each pattern's input nodes inside
`contract` (with the first batch's tensors) and set them itself before each
later call: two code paths for the same inputs. §3.7 has a design that
removes this; it is not implemented.

The same holds for tk's `MPS`: which sites are inputs (`in_features`) is
state of the network, not an argument of `contract` (changing it resets
the network). Its `contract` arguments (`inline_input`,
`marginalize_output`, ...) choose how the same pattern is contracted;
`marginalize_output=True` sums the output sites against a conjugated copy.

### 1.6 `Program` objects

`net(data, *args)` is `net.program(*args)(data)`: `net.program(*args)`
returns the program of those arguments (made, not yet recorded, if it does
not exist). A loop never needs it. `tk.Program(net, fn)` records a function
other than `contract` (no arguments, returns a node or a tuple of nodes);
the caller keeps it, and it is recorded again after a `reset`:

```python
program = tk.Program(net, lambda: (vector @ node).renormalize())
vector.tensor = x       # set the nodes it reads
y = program()           # records the first time, replays afterwards
```

### 1.7 Errors

| situation | error |
|---|---|
| `max_programs` not an int or `None` (or a bool) | `TypeError` |
| `max_programs < 1` | `ValueError` |
| arguments of `contract` not hashable | `TypeError` |
| `Program(net, fn)` with `net` not a `TensorNetwork` or `fn` not callable | `TypeError` |
| `fn` (or `contract`) returns something other than a node or a tuple of nodes | `TypeError` |
| a program called while another program of the network is being recorded | `ValueError` |
| `trace` on a network whose `forward` does not call a program | `ValueError` |

If `fn` raises during the recording, the partial recording is dropped and
the next call records again.


## 2. What happens underneath

### 2.1 One recording per network, before

`forward` recorded **one** sequence of operations: the first call ran
`contract` and appended each operation's `(name, args)` to `net._seq_ops`;
later calls replayed that list while the arguments of `contract` stayed the
same. When they changed, `forward` warned, called `reset` (deleting every
resultant node) and recorded again. A loss with two contractions paid a
reset and a recording twice per step.

### 2.2 Why several recordings could not share a network

- **The successor cache.** An operation stores its result node on its first
  operand, under its arguments (`node._successors[name][args]`), and is
  appended to `_seq_ops` only when it creates that node. Two contractions
  that start with the same operation on the same nodes (`stack` of the same
  cores, `core @ data`) share the entry: the second finds it, computes it,
  and does not record it. Its replay then skips the step and reads what the
  first contraction last computed: wrong values, silently.
- **The inverse memory** (what `trace` uses to free tensors) was keyed by
  node *names*; tk renames `x` to `x_0` when a second `x` appears, and
  the traced replay then failed with a `KeyError`.

### 2.3 What changed

- Each program has its own successor cache, `_seq_ops` and inverse memory.
  Operations reach them through the network (`net._program` is the program
  being called), so no operation changed.
- The inverse memory is keyed by the node that stores the memory.
- `forward(data, *args)` is `self.program(*args)(data)`.

### 2.4 Speed: what replay saves, and what it does not

Replay skips `contract`'s Python and each operation's cache lookup. Each
operation still costs its kernel and the bookkeeping around it (reading the
operands' tensors, setting the result's), which dominate. Measured on the
example's network (n = 12, r = 8, batch 64), forward only:

| | with grad | no grad |
|---|---|---|
| the same nodes kept per contraction, each call through the operations' lookups (what a hand-made plan per pattern does) | 1.42 ms | 1.00 ms |
| programs (replay) | 1.39 ms | 1.01 ms |

So programs are not faster than keeping one's own nodes per contraction and
re-running the operations; they are as fast without the hand-made plans, and
correct by construction (§2.2). On tiny steps (a sampling step, 3
operations, 100 samples, r = 8) the lookup is a larger share: 70 → 57 µs.
The large saving is against the old single recording: 9.1 → 3.3 ms per
training step in §1.2, because nothing is reset and recorded again.


## 3. Design choices

### 3.1 The two designs of the issue

- **(a) Programs keyed by the arguments, inside the network**: `forward`
  looks up the program of its arguments; the public interface gains only
  the number of programs kept.
- **(b) A program object** (`program = net.program(*args)`,
  `program(data)`), which owns its recording and shares the network's
  parameters.

Chosen: both, as one mechanism. The object of (b) holds the state; the
network's dict of (a) maps arguments to objects, and `forward` is
`self.program(*args)(data)`. The old single recording is the one-key case
with `max_programs = 1`, not a second code path. (a) alone needs per-key
dicts of the same state inside the network and has no place for a function
other than `contract`; (b) alone leaves `forward` to the user.

### 3.2 State through the network, not through the operations

The operations (six of them hand-written: `split`, `merge_edges`,
`split_edge`, `stack`, `unbind`, `einsum`) use `net._seq_ops`,
`net._tracing`, `net._traced`, `net._inverse_memory` and
`node._successors`. These became properties that resolve to the program
being called. Rejected: passing the program through every operation (an
argument threaded through every signature), and moving the six operations
onto the core first (larger, and not needed here).

### 3.3 Per-program successor caches

Each node keeps one successor dict per program
(`node._successor_caches[program]`). Rejected: sharing results and
appending the step to the second program when it hits the first one's
cache. That needs the operands of every hit recorded in the second program's
inverse memory, which `Operation.__call__` cannot do (it does not know the
operands), and it couples programs: one traced replay frees a tensor the
other has not recomputed. Cost of isolation: an operation common to two
programs is computed in each.

### 3.4 Freeing all programs at once

Programs are freed only by `reset`; `max_programs = n` resets when the
`n+1`-th is recorded, instead of evicting the least recently used. Evicting
one program deletes some of the network's nodes: without renaming, names
can collide; with renaming, the other programs' nodes are renamed (the
quadratic renaming of `TK_NOTES.md`); and with `auto_stack`, the first
program to stack the parameters owns the `ParamStackNode` the others point
to.

### 3.5 Inputs as nodes

A program reads its nodes' tensors; the network's data nodes are set by
`program(data)` as `forward` did. Considered: a list of input nodes
returned by the function with its outputs (`fn() -> (inputs, outputs)`,
`program(*tensors)`), a second return convention. For per-pattern inputs
see §3.7.

### 3.6 `trace` through the network

`trace` calls the network (`self(example, *args)`), because subclasses
override `forward` and may turn their arguments into other `contract`
arguments; so it marks the program the call ran (`net._last_program`). For
that, every recording counts reads in the inverse memory (before, only
`trace`'s did): a dict update per operand while recording, no effect until
freeing is on.

### 3.7 Open: per-pattern inputs

For §1.5. Inputs that a program creates while it is recorded would be its
own: `contract` creates them with a placeholder of batch size 1 (as
`trace`'s example of zeros), the program records on the placeholders, then
fills them with the call's data and replays (the first call computes twice,
the first time on one sample). `net(data, *args)` fills the program's own
inputs if it has any, else the network's data nodes. A sketch:

```python
def contract(self, observed):            # observed: a tuple of variables
    ...                                  # copies of the cores for this pattern
    for i in observed:
        phi = self.input(shape=(1, d), axes_names=('batch', 'feature'))
        phi['feature'] ^ ket[i]['input']
    ...

def nll(net, x):                         # chain rule over the order 0..n-1
    phi = net.feature_map(x)             # (batch, n, d)
    log_P = [net([phi[:, i] for i in range(k)], observed=tuple(range(k)))
             for k in range(n + 1)]      # n + 1 programs, recorded once
    return -(log_P[n] - log_P[0]).mean()
```

(For this particular loss the sum telescopes to $-\ln P(x) + \ln Z$, two
contractions; the sketch shows the mechanism, which a loss with weights
$lpha_k$ per term, or one over several orders, needs.)

### 3.8 Open: one switch for freeing

`trace(example, *args)` per program needs the list of argument sets in
advance. Since every recording counts reads, freeing could be a network
setting that holds for every program, recorded or not (like `auto_stack`).

### 3.9 `MPS.sample` stays as it is

Its fixed step networks can be programs (`test_program_of_function` is
such a step), but `sample` does not get shorter: each step's expression
becomes a lambda, one for one, and the steps read their output node's axes,
which a program's tensors-only return does not give. Gain: 13 of 70 µs per
step.

### 3.10 Downstream: per-pattern plans

A tree network contracted in many patterns kept one plan per pattern: its
own virtual nodes, wired once (the per-program cache, by hand), a Python
loop over the steps per call, the leaves refilled with the private
`_direct_set_tensor`, operands freed by hand. Programs give the same speed
(§2.4) without the plan class: `contract(pattern)` keeps the construction
(copies, inputs, path), and replay, freeing and the program per pattern come
from tk. Its `contract` takes the observed *values* (`observed`: variable
→ tensor), which as a program key would make a new program per batch: the
port needs the pattern as the argument and the values as inputs, i.e. §3.7.


## 4. Implementation

### 4.1 Where the code lives (`tensorkrowch/components.py`)

| what | where |
|---|---|
| `Program` | end of the module: `_clear`, `__call__` (data, record or replay), `_record` (runs `fn`, checks outputs, marks them not erasable, resolves the steps), `_eager_state` |
| network state | `TensorNetwork.__init__`: `_eager`, `_program`, `_programs` (key → Program), `_max_programs`, `_generation`, `_last_program` |
| properties to the program being called | `_seq_ops`, `_inverse_memory`, `_tracing` (= the program is recording), `_traced` |
| `max_programs`, `_full` | after `resultant_nodes` |
| `program`, `_contract_key`, `_hashable_arg` | after `trace`; `_hashable_arg` is a module function |
| `forward`, `trace`, `reset` | `forward` = `self.program(*args, **kwargs)(data)`; `trace` resets if full, calls the network, marks `_last_program` traced; `reset` empties `_programs`, increases `_generation`, clears `_eager` and every node's `_successor_caches` |
| successors | `AbstractNode._successors` (property): `_successor_caches[network._program]` |
| inverse memory | `_record_in_inverse_memory`, `_check_inverse_memory`: keyed by `node.node_ref()`; `_update_node_info` no longer moves keys |

### 4.2 Call flow

```
forward(data, *args) -> program = net.program(*args)
  _contract_key: bind to contract's signature, defaults, hashable
  found -> program; not found -> reset if full (warning), new Program
program(data)
  data: set_data_nodes if none, add_data
  net._program = program (operations now use its cache and state)
  recorded since the last reset?
    no  -> _record: _recording = True; outputs = fn(); checks;
           outputs not erasable; steps = [(fn_next, check_first(*args), args)]
    yes -> for fn_next, successor, args in steps: fn_next(successor, *args)
  net._program = net._eager
  return the outputs' tensors
```

Outside programs, operations use `net._eager`, a program that is never
called: eager operations keep their own cache, append to its `_seq_ops`,
and neither record nor check the inverse memory.

### 4.3 Behaviour changes

- `forward` returns the tensor of the node `contract` returned, also in the
  replay (before: the last operation's output, which had to be that node).
- Untraced recordings record the inverse memory too (no effect until
  `trace`).
- Eager operations no longer share successors with `forward`'s program, so
  an eager call does not consume or free the program's tensors; and
  `forward` after eager operations records without a warning (before: it
  found resultant nodes, warned and reset).
- `trace` resets only when the network is full (with the default, always).
- Two tests read the removed private attributes `_contract_outputs` and
  `net._traced`; they read `net._programs` and `net.program()._traced` now.

### 4.4 Tests

`tests/test_components.py::TestProgram`:

- two orders of a chain alternate, float64 and complex128: values and
  gradients equal a dense contraction, no warning, the number of resultant
  nodes is constant after the first recording of each;
- three `MPS.contract` variants alternate and equal a fresh copy's;
- `max_programs`: same arguments (normalised) give the same program, the
  `n+1`-th resets with the warning, lowering it resets, type and value
  errors;
- two traced programs each free their intermediates and the data;
- a traced program survives a rename of its nodes;
- `Program(net, fn)`: replays without calling `fn`, with inputs set by
  `node.tensor = t` and another batch size, records again after `reset`;
  errors; a failed recording is recorded again.

With the per-program caches disabled (one shared cache), the alternation
and the freeing tests fail.
