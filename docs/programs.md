# Programs: several recorded contractions per network

**In short.** A loss may contract a network in several ways: the amplitudes
and the norm, or marginals over different sets of observed variables. Set
`net.max_programs = None` once and call the network as usual,
`net(data, *args)`, for each contraction. The first call with each set of
`contract` arguments records it (a *program*); every later call with the
same arguments replays it. Nothing is declared before the loop. Data that
differ between contractions go into *input nodes* that `contract` creates.
`net.free_intermediates = True` (optional) makes every program erase its
intermediate tensors as soon as it has used them: memory, not speed.

Parts: the recipe for a training loop (§1), what happens underneath (§2),
design choices (§3), implementation (§4), and a larger example with the
contraction interface of tpm-tn-benchmark (§5). It builds on
`docs/custom_operations.md` (operations and successors).


## 1. A training loop with several contractions

### 1.1 Example: negative log-likelihood of a Born machine

$-\ln p(x) = -\ln|\psi(x)|^2 + \ln Z$ needs two contractions per step,
both before the first `backward`: the amplitudes $\psi(x)$ of the batch,
and $Z = \langle\psi|\psi\rangle$. A network whose `contract` takes an
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
        ket = [self.copy_node(c) for c in self.cores]
        bra = [self.copy_node(c) for c in self.cores]
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


torch.manual_seed(0)
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

Output: a loss of about 9.0 after 100 steps, and 2 programs (the last
digits vary across machines). The loss and its gradients equal those of a
dense contraction.

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
   `norm`; in §5 `other` and `observed`). Only arguments that change *which*
   contraction is done belong there; the data goes through
   `net(data, ...)`. Arguments are compared after normalisation
   (`net(norm=True)` and `net(None, True)` are the same program); they must
   be hashable (lists, tuples, dicts and sets of hashable objects are turned
   into tuples). A tensor among them is told apart by object: passing the
   same tensor every call is one program, and a different tensor where an
   existing program had one raises a `ValueError`, since values that change
   between calls belong in `data`.
2. **Set `max_programs`** to at least the number of argument sets the loss
   uses (`None`: no limit). With the default, 1, every switch between
   `net(x)` and `net(norm=True)` warns, resets the network and records
   again: 9.1 ms per training step above, against 3.3 ms.
3. **Create the optimizer after the first step** if the network stacks its
   parameters (`auto_stack`, as tk's models do): the first recording
   replaces the parameters of stacked nodes by the stack's. (The examples
   do not stack, so they create the optimizer first.) With several programs
   that is the first step that has recorded all of them: a program recorded
   before the stacking computes its first call on the parameters before
   they were stacked; from its next call it reads them in the stack.
4. **Choose the contraction order well**: the order `contract` uses is what
   is replayed. The zip-up order above keeps (r, r) environments;
   contracting all transfer matrices first would build `n` tensors of r⁴
   entries.
5. **Optionally set `free_intermediates`** (§1.3).

### 1.3 `free_intermediates`: memory, not speed

`forward` replays from the second call in any case. With
`net.free_intermediates = True`, every program erases each intermediate
tensor (of resultant and data nodes) as soon as it has read it for the last
time, so the network holds nothing between calls:

| n = 100, r = 64, batch 256, under `torch.no_grad` | one call | intermediates the network holds afterwards |
|---|---|---|
| off | 27 ms | 64 MB |
| on | 27 ms | 0 MB |

With autograd, the graph keeps what `backward` needs anyway; then it only
drops the network's own references. Without autograd (evaluation,
sampling), it also lowers the peak during the call.

It is one setting for the network: it holds for every program, recorded
before or after it is set, and `reset` keeps it. Every recording counts how
often each tensor is read, so it can be switched on at any time, e.g. after
the first step. `trace(example, *args)` records the program of `args` and
switches it on.

It is off by default because it changes what a network shows after a call:
the tensors of intermediate nodes are `None`. Code that reads them (when
debugging, or an operation called after `forward` on an intermediate node)
would fail, and the gain is memory only.

### 1.4 Inputs

A program reads the tensors its nodes hold when it is called, as a recorded
CUDA graph reads its input buffers. A network takes its data in one of two
ways:

- **data nodes** (`set_data_nodes`, as tk's models do): one set for the
  network, attached to the leaf nodes' input edges once; `net(data, ...)`
  adds `data` to them before the program runs. Good when every contraction
  reads the same variables (§1.1).
- **input nodes** (`self.input_node(shape, axes_names)`, called inside
  `contract`): each program has its own, created while it is recorded, so
  each contraction can read different variables (§1.5).

The parameters are read through any node that shares their memory:
`self.copy_node(node)`, called inside `contract`, makes a copy of a node
that shares its tensor, with new edges, for this program (deleted with it).
A constant input (a summed vector $\sum_x \varphi(x)$, a Gram matrix) is a
plain `tk.Node` of the network made once in `__init__`, and `copy_node` of
it in each `contract` that uses it: no reserved name, nothing passed per
call.

### 1.5 Different observed variables per contraction: input nodes

A marginal $P(x_S)$ observes the variables in $S$ and sums the others. A
variable's leg is connected to its data in one pattern and summed in
another, so each pattern contracts its own copies of the cores, and its
data enters through its own input nodes:

```python
def contract(self, other=None, observed=()):
    ...
    for i in observed:
        rho = self.input_node(shape=(1, d, d),            # placeholder
                              axes_names=('batch', 'ket', 'bra'))
        rho['ket'] ^ ket[i]['input']
        rho['bra'] ^ bra[i]['input']
    ...

net(data, other=net, observed=(0, 2, 3))   # data: one tensor per input node
```

The input node holds zeros of batch size 1. The first call records the
program on these placeholders, then fills them with `data` (a list with
one tensor per input node, in the order `contract` created them) and
replays; later calls fill and replay. §5 is a full example, with a loss
that mixes the likelihood and a marginal likelihood.

### 1.6 `Program` objects

`net(data, *args)` is the interface. Each set of arguments is a
`tk.Program` the network keeps; a loop never handles them.
`tk.Program(net, fn)` records a function other than `contract` (no
arguments, returns a node or a tuple of nodes); the caller keeps it, and it
is recorded again after a `reset`:

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
| `free_intermediates` not a bool | `TypeError` |
| arguments of `contract` not hashable | `TypeError` |
| a tensor among the arguments of `contract` where a program has another tensor (and the same other arguments) | `ValueError` |
| `input_node` or `copy_node` outside a recording; `input_node` in a network with data nodes; `copy_node` of a node of another network | `ValueError` |
| `copy_node` of something other than a `Node` or `ParamNode` | `TypeError` |
| `data` not a list with one tensor per input node, or missing for a program with input nodes | `ValueError` |
| `data` for a program without input nodes, in a network without data nodes | `TypeError` |
| `Program(net, fn)` with `net` not a `TensorNetwork` or `fn` not callable | `TypeError` |
| `fn` (or `contract`) returns something other than a node or a tuple of nodes | `TypeError` |
| a program called while another program of the network is being recorded | `ValueError` |

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
- **The inverse memory** (what freeing uses) was keyed by node *names*; tk
  renames `x` to `x_0` when a second `x` appears, and the freeing replay
  then failed with a `KeyError`.

### 2.3 What changed

- A program is an object that holds one recording: its `_seq_ops`, its
  successor cache, its inverse memory, its outputs and its input nodes. The
  network keeps a dict from (normalised) `contract` arguments to programs,
  and `forward(data, *args)` runs the program of `args`.
- The inverse memory is keyed by the node that stores the memory.

### 2.4 Speed: what replay saves, and what it does not

Replay skips `contract`'s Python and each operation's cache lookup. Each
operation still costs its kernel and the bookkeeping around it (reading the
operands' tensors, setting the result's), which dominate. Measured on §1.1's
network (n = 12, r = 8, batch 64), forward only:

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
- **(b) A program object** (`program(data)`), which owns its recording and
  shares the network's parameters.

Chosen: both, as one mechanism. The object of (b) holds the state; the
network's dict of (a) maps arguments to objects, and `forward` runs the
program of its arguments. The old single recording is the one-key case with
`max_programs = 1`, not a second code path. The lookup of a program by
arguments is private (`_program_for`): users call `forward`. `tk.Program`
stays public for functions other than `contract`.

### 3.2 How operations write into the program being called

`contract` is written with node operations (`@`, `stack`, `conj`,
`contract_between`, custom `tk.operation`s, ...); they are what a program
records. Plain PyTorch on tensors inside `contract` is not recorded (that is
what `tk.operation` is for). While recording, each operation appends itself
to "the recording", caches its result node, and counts its reads; while
replaying, it computes and (with `free_intermediates`) frees.

Before, "the recording" was a set of fields of the network
(`net._seq_ops`, `net._tracing`, `net._traced`, `net._inverse_memory`, and
`node._successors` on the operands), read and written by every operation,
six of which (`split`, `merge_edges`, `split_edge`, `stack`, `unbind`,
`einsum`) do it in their own code. Those names now resolve to the fields of
the program being called (`net._program`): the network sets it for the
duration of a call. So no operation changed beyond two renames that say
what the flags mean now: `_tracing` is `_recording` (the program is being
recorded and counts reads) and `_traced` is `_freeing` (the program erases
tensors after their last read). Rejected: passing the program
to every operation (an argument threaded through every signature), and
rewriting the six operations on the shared core first (larger, and not
needed for this).

### 3.3 Each program computes its own results

An operation's result is cached per program: each node keeps one successor
dict per program (`node._successor_caches[program]`). Two programs never
share a result node, even when they do the same operation on the same
nodes. Rejected: letting the second program use the first one's result and
appending the step to the second program's recording. That needs the
step's operands counted in the second program's inverse memory, which the
generic `Operation.__call__` cannot do (it does not know which arguments are
operands), and it couples programs: one program's freeing would erase a
tensor the other has not recomputed.

What isolation costs is structural, not data-dependent: only an operation
on the *same operand nodes* with the same arguments could be shared. With
per-pattern copies of the cores (which the one-edge-per-axis model needs
anyway, §1.5), two programs never operate on the same nodes, so there was
nothing to share. Mathematically equal subcomputations in different
patterns are recomputed: the left environment of $x_0 \dots x_k$ is
contracted in every marginal that observes those variables. Rescaling does
not change this (each pattern rescales its own environments). For a loss
over all prefixes of an order (the chain rule with weights $\alpha_k$), $n$
prefix programs cost $O(n^2)$ site contractions, where one sweep that keeps
every left environment, and returns every prefix marginal as one of several
outputs of one program, costs $O(n)$. That sharing is the contraction
method's job, not the replay's (§3.9).

### 3.4 Freeing all programs at once

Programs are freed only by `reset`; `max_programs = n` resets when the
`n+1`-th is recorded, instead of evicting the least recently used. Evicting
one program deletes some of the network's nodes: without renaming, names
can collide; with renaming, the other programs' nodes are renamed (the
quadratic renaming of `TK_NOTES.md`); and with `auto_stack`, the first
program to stack the parameters owns the `ParamStackNode` the others point
to.

### 3.5 Input nodes: recorded on placeholders

Per-pattern data needs nodes that exist only in that pattern's
contraction, so `contract` must create them; but `contract` receives only
its arguments, not the data, and a recording computes. So an input node
starts as a placeholder (zeros, batch size 1, the dtype and device of the
network's parameters): the program is recorded on it, then filled and
replayed. The first call computes twice, once on one sample. This is what
`trace` always did: record on an example of zeros, then run.

Considered: the function returning its inputs with its outputs
(`fn() -> (inputs, outputs)`, `program(*tensors)`), a second return
convention; and inputs set by the caller on nodes it keeps (`node.tensor = t`
before each call), which needs the caller to create them during the
recording and keep a registry per pattern: two code paths for the same
inputs.

A network takes its data either in data nodes or in input nodes;
`input_node` raises in a network with data nodes, so `net(data)` always
knows where `data` goes (data nodes if the network has them or overrides
`set_data_nodes`, else the program's input nodes). Input nodes are virtual
nodes: `free_intermediates` does not erase them (they hold the inputs until
the next call), and `reset` deletes them with their program.

### 3.6 Freeing: one setting for the network

`free_intermediates` holds for every program. Rejected: freeing per program
(`trace(example, *args)` once per argument set), which needs the list of
argument sets before the loop. Every recording counts reads (before, only
`trace`'s did): a dict update per operand while recording, no effect until
freeing is on. `trace` records through `self(example, *args)`, because
subclasses override `forward` and may turn their arguments into others
(§5's `forward` turns values into input tensors).

### 3.7 Replay without the lookup

At the end of a recording, each step is resolved once to
`(fn_next, successor, args)`; the replay calls `fn_next` directly (§2.4).

### 3.8 `MPS.sample` stays as it is

Its fixed step networks can be programs (`test_program_of_function` is
such a step), but `sample` does not get shorter: each step's expression
becomes a lambda, one for one, and the steps read their output node's axes,
which a program's tensors-only return does not give. Gain: 13 of 70 µs per
step.

### 3.9 Downstream: per-pattern plans and a deep contraction method

A tree network contracted in many patterns kept one plan per pattern: its
own virtual nodes, wired once (the per-program cache, by hand), a Python
loop over the steps per call, the leaves refilled with the private
`_direct_set_tensor`, operands freed by hand. Programs give the same speed
(§2.4) without the plan class: `contract(pattern)` keeps the construction
(copies, inputs, path), and replay, freeing, per-pattern inputs and the
program per pattern come from tk. §5 is this design on a chain.

Programs make each pattern cheap to *call*; what a pattern computes is up
to `contract`. Sharing work across patterns (all prefix marginals in one
sweep, §3.3; conditionals from cached environments) belongs in a deep
contraction method that returns several outputs from one program.


## 4. Implementation

### 4.1 Where the code lives (`tensorkrowch/components.py`)

| what | where |
|---|---|
| `Program` | end of the module: `_clear`, `__call__` (where the data goes; record, fill inputs, replay), `_fill_inputs`, `_record` (runs `fn`, checks outputs, marks them not erasable, resolves the steps), `_eager_state` |
| network state | `TensorNetwork.__init__`: `_eager`, `_program`, `_programs` (key → Program), `_max_programs`, `_free_intermediates`, `_generation` |
| properties to the program being called | `_seq_ops`, `_inverse_memory`, `_recording`; `_freeing` is a plain attribute that `Program.__call__` sets around its replay (read by every `fn_next`) |
| `max_programs`, `free_intermediates`, `_full` | after `resultant_nodes` |
| `_program_for`, `_contract_key`, `_hashable_arg` | after `trace`; `_hashable_arg` is a module function |
| `input_node`, `copy_node`, `_recording_program`, `_takes_data_nodes` | before `_add_data_or_set_nodes` |
| pickling (`torch.save`) and deep copies | the recording survives, as on `main`: `Successor.__getstate__` (on #3) drops the kernel, a local function, which the core's `fn_next` plans again; `Program.__getstate__` drops `_steps` (they hold the operations' local functions), which `__call__` finds again (`_find_steps`) |
| tensor arguments | `_check_tensor_arguments`, `_mask_tensors`, `_tensor_ids`, when a new program is made |
| references that follow moved memory | `AbstractNode.tensor`, `tensor_address`, `_direct_get_tensor`: a node that references another (`set_tensor_from`, `copy_node`) reads through it when that node's memory moved later (into a stack, with `auto_stack`) |
| `forward`, `trace`, `reset` | `forward` = `self._program_for(*args, **kwargs)(data)`; `trace` resets if full, calls the network, sets `free_intermediates`; `reset` empties `_programs`, increases `_generation`, clears `_eager` and every node's `_successor_caches` |
| successors | `AbstractNode._successors` (property): `_successor_caches[network._program]` |
| inverse memory | `_record_in_inverse_memory`, `_check_inverse_memory`: keyed by `node.node_ref()`; `_update_node_info` no longer moves keys |
| an operand's parameter as a result | `_create_resultant_from_axes` (`operations.py`) turns it into a view |

### 4.2 Call flow

```
forward(data, *args) -> program = net._program_for(*args)
  _contract_key: bind to contract's signature, defaults, hashable
  found -> program; not found -> reset if full (warning), new Program
program(data)
  data: to the data nodes (set_data_nodes if none, add_data) if the network
        takes data nodes; otherwise kept for the program's input nodes
  net._program = program (operations now use its cache and state)
  not recorded since the last reset -> _record:
        _recording = True; outputs = fn() (input_node: placeholders);
        checks; outputs not erasable;
        steps = [(fn_next, check_first(*args), args)]
  input data -> _fill_inputs
  recorded before, or inputs filled -> for each step: fn_next(successor, *args)
  net._program = net._eager
  return the outputs' tensors
```

Outside programs, operations use `net._eager`, a program that is never
called: eager operations keep their own cache, append to its `_seq_ops`,
and neither record nor check the inverse memory.

### 4.3 Behaviour changes

- `forward` returns the tensor of the node `contract` returned, also in the
  replay (before: the last operation's output, which had to be that node).
- Every recording counts reads in the inverse memory (no effect until
  `free_intermediates`).
- Freeing is a setting of the network (`free_intermediates`), set by
  `trace` and kept by `reset` (before: `reset` untraced the network).
- Eager operations no longer share successors with `forward`'s program, so
  an eager call does not consume or free the program's tensors; and
  `forward` after eager operations records without a warning (before: it
  found resultant nodes, warned and reset).
- `trace` resets only when the network is full (with the default, always).
- Fix: `conj` of a real parameter returns the parameter itself, and the
  resultant node detached it, so its first call had no gradient.
- Two tests read the removed private attributes `_contract_outputs` and
  `net._traced`; they read `net._programs` and `net.free_intermediates`.
  Three tests set `net._tracing` and `net._traced` around eager operations
  to check freeing; they are one test, parametrised, through a `Program`
  with `free_intermediates` (end state and values).
- A network pickled or deep-copied after `forward` keeps its programs, nodes
  and parameters, as on `main`; only the local functions (kernels, steps)
  are found again in the next call (`torch.save` failed on them; on #3's
  branch too, fixed there). Training goes on after a load with the same
  parameters and `state_dict` keys. A network with a `tk.Program` whose
  function is a lambda cannot be pickled (the program is a key of its
  nodes' caches), as with any module that holds a lambda; a module-level
  function or a `functools.partial` can.
- A single-element tensor among `contract`'s arguments that changes between
  calls raises (see the tensor arguments above); on `main` it compared to
  one bool, so the network was reset and recorded again.
- A node made with `set_tensor_from` (or `copy_node`) follows the node it
  references when that node's memory moves into a stack later; it used to
  lose it (`tensor_address()` gave `None`, which a test pinned as
  "undesired"; the test now checks the fix). A program of copies recorded
  before a program that stacks the same nodes raised `KeyError` in its
  replay.
- A tensor argument of `contract` other than an existing program's raises
  a `ValueError` (on `main`, comparing the arguments raised a
  `RuntimeError`).
- Eager operations cost about 0.3 µs more than on #3's branch (13.4 vs
  13.1 µs for `a @ b` on small tensors; #3 itself adds about 1.2 µs over
  `main`): the successor lookup through the program being called.

### 4.4 Tests

`tests/test_components.py::TestProgram`:

- two orders of a chain alternate, float64 and complex128: values and
  gradients equal a dense contraction, no warning, the number of resultant
  nodes is constant after the first recording of each;
- three `MPS.contract` variants alternate and equal a fresh copy's;
- `max_programs`: same arguments (normalised) give the same program, the
  `n+1`-th resets with the warning, lowering it resets, type and value
  errors;
- two programs each free their intermediates and the data;
  `free_intermediates` holds for recorded programs, is kept by `reset` and
  set by `trace`;
- a freeing program survives a rename of its nodes;
- §5's `Chain`, three programs with input nodes (Z, the likelihood, a
  marginal), float64 and complex128, freeing on and off: the mixed loss and
  its gradients equal a dense contraction;
- `copy_node`: shares the tensor and the gradient, new dangling edges,
  deleted by `reset`, errors; copies and a stacking `MPS` program
  alternating in both orders, with parameter updates: values and gradients;
- `torch.save`/`torch.load` after a forward (programs with input nodes);
  restored networks (`TestTensorNetwork::test_restored_network_after_forward`
  and `test_restored_nodes_and_results`, on #3, pickled and deep-copied):
  values (also another batch size), `state_dict` keys and parameters right
  after loading, the same nodes, training goes on, a node restored before
  its network, results of operations called by hand; tensor arguments: the
  same object replays, another raises;
- input-node errors (also a call without data); `Program(net, fn)`: replays without calling `fn`, with
  inputs set by `node.tensor = t` and another batch size, records again
  after `reset`; errors; a failed recording is recorded again.

`tests/test_operations.py::test_conj_real_param_grad`: the `conj` fix.

With the per-program caches disabled (one shared cache), the alternation
and the freeing tests fail.


## 5. A larger example: the interface of tpm-tn-benchmark

The contraction interface of tpm-tn-benchmark's `TreeNetwork`:
`contract(other=None, observed=..., open=(), keep_off_diagonal=False)`.
Each variable is observed (contracted with its data), open (left as an
output leg) or summed; `other=None` contracts one layer, `other=net` two
(the bra conjugated), where a summed variable is summed between the
layers, and `other=self` with everything summed is $Z = \langle\psi|\psi\rangle$.
Here on a chain, with `other=self` only and no open variables, and with
every site's contraction rescaled (log-scales returned and summed in log
space), so that no product of many factors overflows.

The pattern (`other`, `observed`) is the program's key; the values are its
data. `forward` turns the batch `x` (all variables) into one input per
observed variable, $\varphi(x_i)\varphi(x_i)^\dagger$ (one-hot $\varphi$),
in the order of `observed`. The loss mixes the likelihood and the marginal
likelihood of the even variables, three programs per step:

$$
L = -\frac{1}{B}\sum_b \Big[\alpha \ln p(x^b) + (1-\alpha) \ln p(x^b_S)\Big],
\qquad p(x_S) = \frac{P(x_S)}{Z}.
$$

```python
import torch
import tensorkrowch as tk


class Chain(tk.TensorNetwork):
    """psi(x) = A_0[x_0] ... A_{n-1}[x_{n-1}], with the contraction interface
    of tpm-tn-benchmark: each variable is observed, summed or (not here) open.

    contract(other, observed): one layer (other=None) or two (other=self,
    the bra conjugated); `observed` is a tuple of variables, whose values
    forward takes in `x`; every other variable is summed. Every site's
    contraction is rescaled, so the result is (value, log-scales) and no
    product of many factors overflows.
    """

    def __init__(self, n, d, r, dtype=torch.float64):
        super().__init__(name='chain')
        self.d = d
        dims = [1] + [r] * (n - 1) + [1]
        self.cores = [tk.randn(shape=(dims[i], d, dims[i + 1]),
                               axes_names=('left', 'input', 'right'),
                               name='core', network=self, param_node=True,
                               std=0.5, dtype=dtype)
                      for i in range(n)]
        for a, b in zip(self.cores[:-1], self.cores[1:]):
            a['right'] ^ b['left']

    def _layer(self):
        """Copies of the cores sharing their memory, wired along the chain."""
        copies = [self.copy_node(core) for core in self.cores]
        for a, b in zip(copies[:-1], copies[1:]):
            a['right'] ^ b['left']
        return copies

    def contract(self, other=None, observed=()):
        if other is not self:
            raise ValueError('this example has only other=self')
        ket, bra = self._layer(), self._layer()
        logs = []
        env = None
        for i, (k, b) in enumerate(zip(ket, bra)):
            if i in observed:   # phi(x_i) phi(x_i)^dagger, an input per call
                rho = self.input_node(shape=(1, self.d, self.d),
                                      axes_names=('batch', 'ket', 'bra'))
                rho['ket'] ^ k['input']
                rho['bra'] ^ b['input']
            else:               # summed: <ket| and |bra> share the variable
                k['input'] ^ b['input']
            half = k if env is None else env @ k
            if i in observed:
                half = half @ rho
            env, log = half.contract_between(b.conj(), rescale=True)
            logs.append(log)
        return (env, *logs)

    def forward(self, x=None, other=None, observed=()):
        """x: (batch, n) values; the observed ones are used."""
        data = None
        if observed:
            phi = torch.nn.functional.one_hot(x, self.d).to(
                self.cores[0].tensor.dtype)
            data = [torch.einsum('bi,bj->bij', phi[:, i], phi[:, i].conj())
                    for i in observed]
        return super().forward(data, other=other, observed=tuple(observed))

    def log_value(self, x=None, other=None, observed=()):
        """log |contraction|: the rescaled value's log plus the log-scales."""
        value, *logs = self(x, other=other, observed=observed)
        return value.abs().log().reshape(-1) + sum(logs)


torch.manual_seed(0)
n, d, r = 20, 3, 8
net = Chain(n, d, r)
net.max_programs = None           # three programs
net.free_intermediates = True     # optional: memory, not speed
optimizer = torch.optim.Adam(net.parameters(), lr=1e-2)

S = tuple(range(0, n, 2))         # the even variables
alpha = 0.7                       # weight of the full likelihood
x = torch.randint(d, (64, n))
for step in range(100):
    optimizer.zero_grad()
    log_Z = net.log_value(other=net)                                 # 1
    log_p = net.log_value(x, other=net, observed=range(n)) - log_Z   # 2
    log_p_S = net.log_value(x, other=net, observed=S) - log_Z        # 3
    loss = -(alpha * log_p + (1 - alpha) * log_p_S).mean()
    loss.backward()
    optimizer.step()
print(f'{loss.item():.3f}', len(net._programs))
```

Output: a loss of about 14.1 after 100 steps (from about 21.1 at the
first), and 3 programs (the last digits vary across machines). The values and gradients equal a dense contraction, in
float64 and complex128 (`tests/test_components.py::TestProgram`, which has
this `Chain`).

What each piece does:

- `_layer` makes copies of the cores that share their memory
  (`copy_node`): each pattern wires its own copies, since a leg can be
  connected to one other leg only. The copies belong to the program and are
  deleted with it.
- `contract` zips the two layers from the left. At site `i`, the ket copy,
  then (if observed) the input $\varphi\varphi^\dagger$, then the
  conjugated bra copy are absorbed into the environment; the last step is
  `contract_between(..., rescale=True)`, which divides the environment by
  its largest entry per sample and returns that scale's log. The program's
  outputs are the environment and the `n` log-scales.
- `log_value` adds the log of the rescaled value to the log-scales.
- In the loop, `net.log_value(other=net)` has no inputs (it is $\ln Z$,
  without batch); the two others have one input per observed variable.

Variants that fit the same scheme: `open` variables (leave their legs
dangling, and return them as outputs: a marginal over them); a dictionary
for `observed` (turn it into a tuple of variables for the key and a list
of values for the data in `forward`); conditionals
$p(x_I \mid x_J) = P(x_{I\cup J})/P(x_J)$, two programs.

`other` a different network: its nodes cannot be connected to this
network's (`copy_node` and `set_tensor_from` work within one network), and
a copy of its tensors goes stale after an optimizer step. What works is
passing `other`'s tensors as inputs: `contract` creates one input node per
core of `other` (`input_node`, the bra layer), and `forward` passes
`[core.tensor for core in other.cores]` in `data`, with the observed
values. The review of #4 checked values and gradients (also with respect to
`other`'s parameters) against a dense contraction this way.
