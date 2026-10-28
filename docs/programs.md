# Programs: several recorded contractions per network

A `Program` is a sequence of operations recorded from one call of a function
on a network's nodes and replayed in the next calls. `forward` is a program
of `contract`, one per set of `contract` arguments, so a network can be
contracted in several ways without being reset between them (issue #2).
This document has four parts: primer, user guide, design choices,
implementation. It builds on `docs/custom_operations.md` (operations,
successors, `trace`).


## 1. Primer

### One recording per network, before

`forward` recorded **one** sequence of operations: the first call ran
`contract` and appended each operation's `(name, args)` to `net._seq_ops`;
the next calls replayed that list while the normalised arguments of
`contract` stayed the same. When they changed, `forward` warned
("Arguments of contract have changed…"), called `reset` and recorded again.

Models used in more than one way need many contractions over the same
parameters: log Z, likelihoods with any subset observed, marginals,
sampling steps. Each switch threw away the previous recording and its
nodes, so code that needed several contractions kept its own nodes per
pattern and drove the operations by hand, with every call paying each
operation's cache lookup (`check_first`).

### Why several recordings could not share a network

Two things were global to the network:

- **The successor cache.** An operation stores its result node on its first
  operand, under its arguments (`node._successors[name][args]`), and appends
  to `_seq_ops` only in `fn_first`. Two contractions that start with the same
  operation on the same nodes (e.g. `node @ data` in both, or `stack` of the
  same leaves) share that entry: the second one finds it, runs `fn_next`, and
  never records the step. Replaying the second contraction then skips the
  step and reads whatever the first one last computed.
- **The inverse memory**, keyed by node *names*: a node renamed after
  tracing (tensorkrowch renames `x` to `x_0` when a second `x` appears)
  made the traced replay fail with a `KeyError`.

### What changes

- Each program has its own successor cache, `_seq_ops` and inverse memory.
  Operations reach them through the network (`net._program` is the program
  being called), so operations are unchanged.
- The inverse memory is keyed by the node that stores the memory, not by
  its name.
- `forward(data, *args)` is `self.program(*args)(data)`.


## 2. User guide

### 2.1 Several contractions of one network

```python
mps = tk.models.MPS(n_sites=6, phys_dim=3, bond_dim=4,
                    in_features=[0, 1, 3, 4])
mps.max_programs = None          # keep every program until reset

for data in batches:
    amplitudes = mps(data)                           # program 1
    marginal = mps(data, marginalize_output=True)    # program 2
```

The first call with each set of arguments records it; later calls replay
it. No warning, no reset, and the number of nodes stays constant.

`max_programs` (default `1`) is how many programs the network keeps:

| `max_programs` | a new set of arguments |
|---|---|
| `1` (default) | resets the network with the warning, as before |
| `n` | is recorded while there are fewer than `n` programs; then the network is reset (warning) and it is recorded |
| `None` | is always recorded; programs are freed by `reset` |

Setting `max_programs` below the number of programs resets the network.
Programs are only freed together (by `reset`); see §3 for why.

### 2.2 `program` and `Program`

`net.program(*args, **kwargs)` returns the program of
`contract(*args, **kwargs)`; `program(data)` adds `data` to the data nodes
and runs it, which is what `forward` does. A program of any function:

```python
program = tk.Program(net, lambda: (vector @ node).renormalize())
vector.tensor = x       # inputs: set the nodes the program reads
y = program()           # first call records, next calls replay
```

`fn` takes no arguments and returns a node or a tuple of nodes; a call
returns their tensors. `Program(net, fn)` objects are held by the caller,
not by the network, and are recorded again in their first call after a
`reset`.

### 2.3 Inputs

A program reads the tensors its nodes hold when it is called, as a recorded
CUDA graph reads its input buffers. So the inputs of a program are nodes:

- the network's `data` nodes, set by `program(data)` / `forward(data)`;
- the parameters (`leaf` nodes);
- any other node whose tensor is set before the call: virtual nodes that
  `fn` (or `contract`) creates as inputs, set with `node.tensor = t`, or
  data nodes created directly (`Node(..., data=True)`), which a traced
  program frees after their last use.

A program never reads results of another program unless its function uses
them; then it reads what that program last computed.

### 2.4 `trace`

`trace(example, *args)` traces the program of `args`: from then on, each
call frees intermediate tensors after their last use. If the network keeps
`max_programs` already, it is reset first (with the default, always, as
before). Otherwise the other programs stay, so several programs can be
traced:

```python
net.max_programs = None
net.trace(example, True)
net.trace(example, False)    # both traced
```

### 2.5 Errors

| situation | error |
|---|---|
| `max_programs` not an int or `None` (or a bool) | `TypeError` |
| `max_programs < 1` | `ValueError` |
| arguments of `contract` not hashable (after turning lists, tuples, dicts and sets into tuples and frozensets) | `TypeError` |
| `Program(net, fn)` with `net` not a `TensorNetwork` or `fn` not callable | `TypeError` |
| `fn` (or `contract`) returns something other than a node or a tuple of nodes | `TypeError` |
| a program called while another program of the network is being recorded | `ValueError` |
| `trace` on a network whose `forward` does not call a program | `ValueError` |

If `fn` raises during the recording, the partial recording is dropped and
the next call records again.


## 3. Design choices

### The two designs of the issue

- **(a) Programs keyed by the arguments, inside the network**: `forward`
  looks up the program of its arguments; nothing new in the public
  interface but the number of programs kept.
- **(b) A program object** (`program = net.program(*args)`,
  `program(data)`), which owns its recording and shares the network's
  parameters.

Chosen: both, as one mechanism. The object of (b) is the unit of state;
the network's dict of (a) maps arguments to objects, and `forward` is
`self.program(*args)(data)`. Today's single program is the one-key case
with `max_programs = 1`, not a second code path. (a) alone would have
needed per-key dicts of the same state inside the network, and a program of
a function other than `contract` (the sampler's steps) would have had no
place; (b) alone would have left `forward` to the user.

### Where the state lives: through the network, not in the operations

The operations (six of them hand-written: `split`, `merge_edges`,
`split_edge`, `stack`, `unbind`, `einsum`) read and write `net._seq_ops`,
`net._tracing`, `net._traced`, `net._inverse_memory` and
`node._successors`. These became properties that resolve to the program
being called (`net._program`). Rejected: passing the program through every
operation (an argument threaded through every signature), and moving the
six operations onto the core first (a larger change that #2 does not need;
see `custom_operations.md` §4.7).

### Per-program successor caches

Each node keeps one successor dict per program
(`node._successor_caches[program]`); `node._successors` is the one of the
program being called. Rejected: sharing results between programs and
appending a step to the second program when it hits the first one's cache.
That needs the operands of every hit to be recorded in the second
program's inverse memory (the operations' `operands` are not known in
`Operation.__call__`), and it couples programs: one program's traced replay
frees a tensor the other one has not recomputed yet. Cost of isolation: a
prefix common to two programs is computed in each (as two hand-made
plans did).

### Freeing: all programs at once

Programs are freed only by `reset`, and `max_programs = n` resets when the
`n+1`-th program is recorded, instead of evicting the least recently used
one. Evicting one program means deleting some of the network's nodes:

- with `move_names=False`, enumerations get gaps and the next node with the
  same name can collide with an existing one;
- with `move_names=True`, the other programs' nodes are renamed (the
  quadratic renaming of `TK_NOTES.md`);
- with `auto_stack`, the first program to `stack` the leaves owns the
  `ParamStackNode` that the leaves and the other programs' stacks point to;
  deleting it would need the leaves' tensors restored and the other
  programs' stacks re-pointed.

A reset avoids all three. The integer stays (rather than a bool) because a
bound on the number of programs is the memory statement users need.

### Inputs as nodes, not as an argument of the program

The issue asks for per-program inputs, set per call. A program's own data
nodes, set by `program(data)`, were considered: `contract` would create
them during the recording, but the recording runs on real tensors, so the
first call's data would have to reach `contract`, which takes only its
arguments. The other variant, a list of input nodes that the function
returns with its outputs (`fn() -> (inputs, outputs)`, then
`program(*tensors)`), adds a second return convention and a second way to
pass data. Chosen: a program reads its nodes' tensors, so its inputs are
the nodes it reads. They are set with the public `node.tensor = t`; for
the network's data nodes, `program(data)` calls `add_data` as `forward` did.
Costs: `node.tensor = t` is about 3.7 µs against 0.2 µs for the private
`_direct_set_tensor` (fork issue #1's measurements).

### `trace` without the key

`trace` calls the network (`self(example, *args)`), because subclasses
override `forward` and may turn their arguments into other `contract`
arguments; so `trace` cannot compute the program's key itself. It marks
the program that the call ran (`net._last_program`). For that, every
recording now records the inverse memory (before, only the recording made
by `trace` did); this costs a dict update per operand while recording, and
changes nothing until `trace` makes the program free its tensors.

### Replay without the lookup

At the end of a recording, each step is resolved once to
`(fn_next, successor, args)`; the replay calls `fn_next` directly. Before,
the replay called each `Operation` (its `check_first`, then `fn_next`).
Measured on a 3-operation step like the sampler's (100 samples, bond 8):
70 µs per step through `check_first`, 57 µs replayed, 63 µs replayed with
the input set by `node.tensor = t`.

### `MPS.sample` stays as it is

Its fixed step networks can be programs (one `Program` per step key, its
leaves set before each call; `test_program_of_function` is such a step).
It is not simpler: each step's contraction expression becomes a lambda,
one for one; the steps read their output node's axes (to permute the
environment) and renormalize it, which a program's tensors-only return
does not give; and the gain is about 13 µs per step. So `sample` keeps
`_sample_kernel`.

### Downstream: per-pattern plans

A tree network contracted in many patterns (log Z, likelihoods with any
subset observed, marginals) kept one plan per pattern: its own virtual
nodes wired once, a Python loop over the steps per call through each
operation's lookup, the leaves refilled with `_direct_set_tensor`, and
operands freed by hand with `_save_in_network(None)`. With programs:

```python
def contract(self, pattern):      # the plan's construction, unchanged:
    ...                           # copies sharing the parameters
    return log_abs, phase         # (set_tensor_from), φ and Σφ inputs, path

def log_prob(self, x, pattern):
    for node, phi in zip(self._inputs[pattern], self.feature_map(x)):
        node.tensor = phi         # the pattern's input nodes
    return self.program(pattern)()
```

The replay loop, the leaf refills (parameters are read through the copies'
shared memory) and the manual freeing (`trace`) go away; the plans reduce
to a map from pattern to program, which the network keeps
(`max_programs = None`).


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
