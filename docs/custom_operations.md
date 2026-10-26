# Custom operations

`tk.operation` turns a function written in PyTorch into a TensorKrowch
operation. TensorKrowch then records it, caches its resultant nodes and
replays it in `forward` like any built-in operation. This document has four
parts:

1. **Primer**: how TensorKrowch runs a contraction, and why plain PyTorch code
   inside `contract` used to be lost.
2. **User guide**: how to write and use custom operations.
3. **Design choices**: what was considered and why the API looks the way it
   does.
4. **Implementation**: where the code lives and how it interacts with `trace`,
   `reset` and `forward`.


## 1. Primer

### Operations, successors and resultant nodes

Every node operation (`permute`, `mul`, `renormalize`, `contract_edges`,
`stack`, `einsum`, ...) is an `Operation` made of three functions:

| function | when it runs | what it does |
|---|---|---|
| `check_first(*args)` | every call | looks up a `Successor` in `args[0]._successors[name][args]` |
| `fn_first(*args)` | first call with these arguments | computes the result, creates the **resultant node(s)**, stores a `Successor` (the operands' memory references, the children, and *hints* for later calls) and appends `(name, args)` to `net._seq_ops` |
| `fn_next(successor, *args)` | every later call | reads the operands' tensors through the successor, computes the result and replaces the children's tensors; no node, edge or name is created |

What stays the same between calls (permutations, axis numbers, shapes of
reshapes, `opt_einsum` paths) is computed once in `fn_first` and kept as
hints. `fn_next` only does tensor arithmetic.

### `forward`, `trace` and `reset`

- **`forward`.** If the network has no resultant nodes, `forward` calls
  `contract`, which runs its Python code and every operation's `fn_first`.
  Once resultant nodes exist, `forward` **does not call `contract`**: it
  replays `net._seq_ops` in order (`TensorNetwork.operations[name](*args)`,
  i.e. every `fn_next`) and returns the tensor of the last operation's output.
- **`trace`** resets the network and calls `forward` twice. The first call
  runs with `_tracing = True` and records, in `_inverse_memory`, how many times
  each tensor is read and whether it may be freed. Only resultant and data
  tensors may be freed. After that, each `fn_next` frees an intermediate
  tensor after its last read, so traced contractions keep little memory.
- **`reset`** deletes the resultant nodes, empties the successors and forgets
  `_seq_ops`. It is needed only when the structure of the contraction changes;
  `forward` also resets when the arguments of `contract` change.

### The consequence

Only computations made **inside operations** survive the replay. This
`contract` is silently wrong from the second call on:

```python
def contract(self):
    result, log_norm = self.mats[0], 0.
    for node in self.mats[1:]:
        result @= node                         # operation: replayed
        log_norm += result.norm().log()        # plain PyTorch: runs only once
        result = result.renormalize()          # operation: replayed
    return result                              # (log_norm cannot be returned)
```

This matters wherever a contraction is renormalized to avoid overflow and the
scale must be kept. One example is computing Born-machine probabilities,
$\log p(x) = 2\log|\psi(x)| - \log Z$, with a long MPS: in float32,
$|\psi(x)|$ overflows after a few hundred sites, but $\log|\psi(x)|$ does not.
`renormalize` divides by the norm and discards it. Before this change, the
only way out was to rewrite the operation's three functions, or to call
`reset` before every `forward` and give up the replay.


## 2. User guide

### 2.1 A first operation

```python
import torch
import tensorkrowch as tk

@tk.operation
def scale(node, factor=2.):
    return (lambda tensor: factor * tensor), node.axes

node = tk.randn((2, 3), axes_names=('left', 'right'))
result = scale(node)              # resultant node named 'scale'
scale(node, factor=2.) is result  # True: cached like any operation
```

The decorated function returns two things:

- a **kernel**: a function from the operands' tensors to the output
  tensor(s). It can be any PyTorch code;
- the output's **axes**, which decide its axis names and edges (§2.4).

### 2.2 What runs when

```python
@tk.operation
def renormalize_log(node, axis):
    # Runs ONCE, on the first call (and again after reset/trace):
    # resolve everything that does not depend on tensor values
    axis_num = node.get_axis_num(axis)
    other_axes = [ax for ax in node.axes if ax.num != axis_num]

    # Runs on EVERY call, also in forward's replay
    def kernel(tensor):
        norm = tensor.norm(dim=axis_num, keepdim=True)
        norm = torch.where(norm == 0., 1., norm)
        return tensor / norm, norm.log().squeeze(axis_num)

    return kernel, (node.axes, other_axes)   # two output nodes
```

The decorated function is the operation's `fn_first`, and the kernel it
returns is its `fn_next`.

- Code **before** `return` (the *planner*) sees nodes and runs once per
  structure. Use it for axis lookups, permutations and shapes that do not
  change. Do not read tensor values there: they are frozen at the first call.
- Code **inside** the kernel sees tensors and runs every time. Control flow
  on tensor values (`torch.where`, even Python `if` on a tensor) is fine
  there, because the kernel is re-executed.
- The kernel receives the tensors of **all node arguments**, in the order of
  the parameters, including nodes inside lists or tuples.

### 2.3 Using it in a network, with several outputs

`contract` may return a tuple of nodes. `forward` then returns a tuple of
tensors, on the first call, in the replay and after `trace`.

```python
class LogNormMPS(tk.models.MPS):
    def contract(self):
        mats_env = self._input_contraction(
            nodes_env=self.in_env,
            input_nodes=[node.neighbours('input') for node in self.in_env])

        result, log_norm = self._left_node, None
        for node in mats_env:
            result @= node
            result, log_n = renormalize_log(result, 'right')
            log_norm = log_n if log_norm is None else log_norm + log_n
        result @= self._right_node
        return result, log_norm

mps = LogNormMPS(n_sites=784, phys_dim=2, bond_dim=20)   # obc
mps.trace(torch.zeros(1, 784, 2))
result, log_norm = mps(data)               # shapes (batch,) and (batch,)
log_psi = result.abs().log() + log_norm    # finite, also in float32
```

- `log_norm + log_n` is the built-in `add` between nodes, so the
  accumulation is replayed too.
- Nodes returned by `contract` are never freed by the traced replay, even if
  later operations read them.
- For Born-machine probabilities, `log Z` is $2\log\|\psi\|$:
  `MPS.norm(log_scale=True)` returns $\log\|\psi\|$. `norm` removes the data
  nodes, so compute it on a separate MPS sharing the tensors, or `reset` the
  network around it.
- Post-processing that only uses `forward`'s outputs (`result.abs().log()`
  above) does not need to be an operation: it runs on plain tensors outside
  the replay.

### 2.4 Output axes

`axes` describes the resultant node. Each element is one of:

| element | meaning |
|---|---|
| an `Axis` of one of the operand nodes | the output axis takes its name and **its edge**, so the result stays connected like the operand. The size of that dimension must not change. |
| a `str` | a new axis with a new dangling edge (e.g. a dimension the kernel creates or reshapes) |

For one output, `axes` is a sequence of such elements (`node.axes`, `[]` for
a scalar node). For several outputs, it is a sequence of such sequences, one
for each tensor the kernel returns.

If the first axis is the `"stack"` axis of a (Param)StackNode, the output is a
`StackNode`, so it can be `unbind`-ed. Then every axis must come from
(Param)StackNodes.

```python
@tk.operation
def project(node, matrix, axis):
    axis_num = node.get_axis_num(axis)
    axes = node.axes[:axis_num] + ['projected'] + node.axes[axis_num + 1:]

    def kernel(tensor):
        return tensor.movedim(axis_num, -1).matmul(matrix).movedim(-1, axis_num)

    return kernel, axes
```

### 2.5 Arguments and caching

- The resultant nodes are cached under the **bound arguments**: defaults are
  applied and keyword arguments are matched to parameters. Lists are turned
  into tuples (the planner receives tuples). So `op(node, axis='right')` and
  `op(node, 'right')` share one result.
- Arguments other than nodes are **recorded at the first call and reused in
  every replay**, as for every TensorKrowch operation. They must be hashable.
  Tensors are hashed by identity: a fixed tensor (a parameter, a constant
  matrix) is fine, but a different tensor passed later is not seen until the
  network is reset. Pass changing values as nodes.
- `*args` and `**kwargs` are not allowed: every argument has to be recorded.

### 2.6 Names

- The name is the function's name, or `@tk.operation(name='...')`. Resultant
  nodes take that name, and are enumerated (`name_0`, `name_1`) when there
  are several outputs.
- The name cannot be the name of a built-in operation, and leaf nodes cannot
  be renamed to it.
- Defining the same function again (re-running a notebook cell) is allowed.
  After that, only the newest definition can be called: an older one raises a
  `ValueError`. This also covers factories that define several operations
  with the same name: two operations with one name would share cached
  results.
- A function from another module with the same name raises a `ValueError`
  at definition time. Pass `name=` to tell them apart.

### 2.7 Errors

| when | error |
|---|---|
| name of a built-in, blank spaces in the name, name already used by another module | `ValueError` at definition |
| `name` not a `str`, `fn` not callable or nameless without `name`, `*args`/`**kwargs` | `TypeError` at definition |
| no node among the arguments, unhashable argument | `TypeError` at call |
| nodes in different networks | `ValueError` at first call |
| `fn` does not return `(kernel, axes)`, malformed `axes` | `TypeError` at first call |
| an `Axis` of a node that is not an operand, wrong number of output tensors, wrong number of dimensions, changed size of an inherited edge | `ValueError` at first call |
| kernel returns something that is not a tensor | `TypeError` at first call |

Later calls do not check anything, for speed. If a kernel returns tensors of
another shape in a later call, the nodes take that shape.

### 2.8 Limitations

- Operations are not in place (no `permute_`-like variants), and they cannot
  rewire edges beyond inheriting them or creating dangling ones.
- Planner code runs once per structure. Results that depend on tensor values
  belong in the kernel.


## 3. Design choices

### Requirements

1. Users write arbitrary PyTorch, not a composition of built-ins.
2. The replay in `forward` stays a sequence of `fn_next` calls: no Python of
   `contract` runs in a traced forward.
3. Values that are not "the result" (norms, log-norms) can flow between
   operations and out of `forward`.
4. Nothing changes for existing code: built-ins, models, and the semantics of
   `trace` and `reset`.
5. A deep module (Ousterhout): a small interface that hides the bookkeeping
   every operation currently writes by hand.

### Options considered

| option | idea | verdict |
|---|---|---|
| **A. `@tk.operation`** | lift a planner and its kernel into an `Operation` | **chosen** |
| B. a deeper `renormalize` | `renormalize(..., return_norm=True, log_scale=True)` returns the norm as a node | it works, at no cost, but the core stays closed: every new need (another norm, a running max, a gauge factor) needs a maintainer change. Prototyped on `operation-exploration` and dropped |
| C. automatic capture | record every torch call made inside `contract` (e.g. with `__torch_function__`) and replay it | no API at all, but replay is silently stale when Python control flow depends on tensor values (`.item()`, `if`), in-place updates are hard to track, and it duplicates TensorKrowch's memory handling: node tensors live in the network's memory, not in the recorded tensors |
| D. subclassing `Operation` | template methods `plan` / `compute` / `outputs` | the same mechanism as A, with more ceremony per operation |

A and B were both implemented and passed the same MPS checks with the same
timing (§4.5). A was chosen because it removes the cause (a closed set of
operations) rather than one symptom.

### The planner/kernel closure

TensorKrowch already splits every operation into "compute once" (`fn_first`
and its hints) and "compute every time" (`fn_next`). The closure exposes
exactly that split with one function: what the planner computes is captured
by the kernel and becomes its hints. The alternative, two functions with an
explicit hints object, would make users name and pass state that a closure
carries for free.

### Outputs are nodes

Extra values such as log-norms are **resultant nodes**, not side-channel
tensors or network-level accumulators. This reuses everything that already
exists for nodes:

- caching via successors;
- freeing via the inverse memory;
- arithmetic via `add`, `mul` and the rest;
- `reset`.

The dataflow stays explicit in `contract`. A network-level accumulator (e.g.
`net.log_norm` filled by operations) would hide state and need its own reset
and trace rules.

This needed `forward` to return several nodes. `contract` returning a tuple
is detected at the first call. The nodes are recorded, read again after each
replay, and marked as not freeable while tracing. A single node keeps the old
behaviour (the last operation's output).

### Output axes as `Axis` objects or names

Taking `Axis` objects from the operands says which edge each output axis
inherits without new vocabulary. Inherited edges must keep their size, since
the edge belongs to the operand. A `str` covers new dimensions. The rule for
`StackNode` outputs follows from the `Axis` invariant: only stack nodes have a
`"stack"` axis, and it must come first.

### Arguments as the cache key

Built-ins key successors on their arguments. Custom operations do the same,
after binding to the signature, so that equivalent calls share results.
Recording non-node arguments for the replay is what every operation already
does.

One built-in exception is not exposed: `mul`, `div`, `add` and `sub` key on
`(node1,)` when the second operand is a tensor or number, so that one result
is reused whatever the operand, and they pass that operand to the kernel in
every call. The private core supports this (`key` and `runtime` in
`_build_operation`, §4.1). For a user operation, a tensor argument is part
of its identity.

### One core for built-in and custom operations

`permute`, `tprod`, `mul`, `div`, `add`, `sub`, `renormalize` and `conj` are
built with the same core as `tk.operation`: each one is its planner. Their
public functions, cache keys, node names and errors are unchanged. Two
things follow from sharing the core:

- **One place for the bookkeeping.** Each of these operations used to write
  `check_first`, `fn_first` and `fn_next` by hand (about 100 lines each).
  This also removed a bug: `tprod` passed both operands' references to
  each operand's inverse-memory check, so any traced network using it failed.
- **One rule for stacks.** The output is a `StackNode` whenever its first
  axis is the `"stack"` axis. Before, every one of these operations except
  `renormalize` raised a `ValueError` on a StackNode.

### Names and redefinition

The registry `TensorNetwork.operations` and the successors are keyed by name,
because `_seq_ops` replays by name. Two live operations with one name would
silently serve each other's cached results. The rules in §2.6 turn every such
case into an error, while still allowing a notebook cell to be re-run.

### Name of the decorator

`tk.operation` registers a `tk.Operation`, as `torch.library.custom_op`
registers a custom operator. The closeness to `Operation` is deliberate.

### Error handling

Errors follow TensorKrowch's conventions: `TypeError` for wrong types,
`ValueError` for wrong values and conflicting options, messages naming the
argument in backticks. They are checked at definition or at the first call,
never in the replay.


## 4. Implementation

### 4.1 Where the code lives

| what | where |
|---|---|
| core | `_build_operation(name, fn, key=None, runtime=())`, after the `Operation` class: builds `check_first`, `fn_first`, `fn_next` and registers the `Operation` |
| decorator | `operation`, section *CUSTOM OPERATIONS*: validation, signature binding, the redefinition rules, then `_build_operation` |
| built-ins built with the core | `permute`, `tprod`, `mul`, `div`, `add`, `sub` (shared `_elementwise_plan`, `_elementwise_key`), `renormalize`, `conj` |
| argument helpers | `_freeze_arg` (lists to tuples), `_operand_nodes` (nodes in arguments, also nested), `_runtime_values` |
| built-in names | `_BUILTIN_OPERATIONS`, a snapshot of `TensorNetwork.operations` taken after all built-ins are defined |
| custom names | `_CUSTOM_OPERATIONS`: name → (module, qualname) |
| output nodes | `_create_resultant_from_axes(axes, name, network, tensor)` |
| several outputs | `TensorNetwork.forward`, `_contract_outputs` (reset by `reset`) |
| export, docs | `tensorkrowch/__init__.py`, `docs/operations.rst` |

### 4.2 Call flow

```
user call: op(node, 'right')
  wrapper            bind to signature, defaults, lists -> tuples, hash check,
                     check this is the current definition of the name
  Operation.__call__(*args)
    check_first      nodes = _operand_nodes(args); look up
                     nodes[0]._successors[name][args]
    -- not found --> fn_first(*args)
                       same network?; kernel, axes = fn(*args)
                       validate axes; tensors = kernel(node.tensor, ...,
                                                       *runtime values)
                       validate tensors; children = _create_resultant_from_axes
                       Successor(node_ref, index, child,
                                 hints={'kernel', 'single'})
                       store successor; net._seq_ops.append((name, args))
                       if tracing: record operands in the inverse memory
    -- found -----> fn_next(successor, *args)
                       tensors = kernel(_direct_get_tensor(ref, index), ...,
                                        *runtime values)
                       child._direct_set_tensor(tensor) for each child
                       if traced: _check_inverse_memory for each operand

replay in forward:   TensorNetwork.operations[name](*args)
                     -> Operation.__call__ -> check_first -> fn_next
                     (the wrapper is skipped)
```

The planner `fn` runs only inside `fn_first`. The replay calls neither the
wrapper nor the planner. The check and the stored key use `key(*args)` when
`key` is given (`mul`, `div`, `add`, `sub`). Runtime values are the arguments
at the positions in `runtime` that are not nodes (the tensor or number in
`node * tensor`). Built-ins call the core's `Operation` directly; their
public functions turn lists of axes into tuples.

### 4.3 `forward` with several outputs

- **First call** (no resultant nodes): if `contract` returns a tuple or list,
  every element must be a node (`TypeError` otherwise). They are stored in
  `_contract_outputs`. While tracing, their addresses in `_inverse_memory` get
  `erase = False`. `forward` returns their tensors.
- **Replay**: after the replay, if `_contract_outputs` is set, `forward`
  returns their tensors; otherwise it returns the last operation's output, as
  before.
- `reset` sets `_contract_outputs = None`.

### 4.4 `_create_resultant_from_axes`

It creates the resultant node with fresh edges, then replaces the edges of
inherited axes with `net._remove_edge` and `node._add_edge`, the same way
`split_edge` does. New axes keep their dangling edges, which `reset` removes
with the node. When every axis is inherited, it passes the edges to the
constructor directly. With a leading `"stack"` axis it creates a
`StackNode` from the operands' `StackEdge`s. It raises a `ValueError` if the
tensor's number of dimensions, or the size of an inherited non-batch edge,
does not match. Batch edges admit any size.

### 4.5 Tests and measurements

- `tests/test_operations.py::TestResultantFromAxes`: inherited and new axes,
  scalar output, stacks, errors.
- `tests/test_operations.py::TestCustomOperation`: single and several
  outputs, caching and argument binding, lists of nodes, tensor arguments,
  new axes, stacks with `unbind`, every error, redefinition rules. Also a
  network with `stack`, `unbind` and custom operations against `einsum`, in
  the four `auto_stack`/`auto_unbind` modes, before and after `trace`, with
  gradients and `reset`.
- `tests/test_components.py::TestTensorNetwork::test_forward_several_outputs*`:
  tuple and list outputs, outputs reused by later operations (fails without
  the protection from freeing), gradients, reset, errors.
- `tests/models/test_mps.py::TestOperationLogNormMPS` (checks in
  `LogNormMPSChecks`), on an MPS subclass that accumulates log-norms with a
  custom operation:
  - $\log|\psi|$ and its gradients equal those of `tk.models.MPS`, in
    float64, before and after `trace`;
  - with 200 sites in float32, where `tk.models.MPS` returns `nan`, it is
    finite and equal to the float64 reference.
- Instrumented: the planner runs once per structure (once at the first call,
  once in `trace`), the kernel on every call. A traced `forward` is the
  replay of `contract_edges` and the custom operation only.
- `tests/test_operations.py::TestTensorOps::test_tensor_ops_stack`: the eight
  built-ins on a StackNode give StackNodes whose unbound nodes are the
  per-node results. `test_tprod_in_forward`: `tprod` in a traced network.
- Built-ins on the core: `MPS.forward` with `renormalize=True` and
  `inline_mats=True` (200 `renormalize` and 200 contractions per call),
  traced, one thread, interleaved rounds: 10.7 ms before and 10.9 ms after,
  within the ~1 ms noise between rounds.
- Timing of the log-norm operation (CPU, 4 threads, 100 sites, bond 20,
  batch 256, traced, medians of two rounds): plain `renormalize` (norm discarded) 17.9 / 57.8 ms
  (inference / forward + backward); custom operation keeping the log-norm
  18.6–18.8 / 58.0–61.2 ms, within noise. The B prototype took
  17.8–17.9 / 58.1–58.3 ms.

### 4.6 Known issues and possible next steps

- Creating a leaf node whose name is an operation name (built-in or custom),
  e.g. `tk.randn((2,), name='renormalize')`, raises a `RecursionError` in
  `TensorNetwork._assign_node_name` instead of its `ValueError`. Renaming a
  node raises the `ValueError` correctly. This predates this change.
- The node-like built-ins (`split`, `contract_edges`, `merge_edges`,
  `split_edge`, `stack`, `unbind`, `einsum`) still write their own
  bookkeeping. They create several nodes with new connected edges, or share
  memory between nodes, which the core does not describe yet.
- A shorthand for the common "same axes as the first node" output could be
  added if it proves frequent.
