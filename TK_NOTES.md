# Notes on tensorkrowch internals (from bm4tc's sampling work, 2026-10-14)

Written while porting `MPS.condition`/`MPS.sample` between tensorkrowch 1.1.6 (what
bm4tc pins, as `SamplingMPS` in `bm4tc/core/sampling.py`) and this fork's `develop`,
and while removing `reset` from bm4tc's forward. Line numbers are tk 1.1.6
(`site-packages/tensorkrowch`), unless marked develop. Everything here was checked in
the source or by a probe; guesses are marked as such.

## Operations, successors and the Python-level graph

- Every node operation is an `Operation` (`operations.py:68`) with `check_first`,
  `fn_first`, `fn_next`. `check_first` looks up a `Successor` in the **first
  operand's** `_successors[op_name]`, keyed by the operation's arguments (e.g.
  `(node, tuple(axes))` for `permute`, `(None, node1, node2)` for `contract_edges`).
- `fn_first` builds the result node (`Node._create_resultant`: name, edges, network
  registration), computes hints (permutations, shapes) and stores the `Successor`
  (`node_ref`, `index`, `child`, `hints`). It also appends `(op_name, args)` to
  `net._seq_ops` — **only `fn_first` appends**, so `_seq_ops` does not grow on reuse.
- `fn_next` reads the operands' tensors through the stored `node_ref`/`index`,
  runs the torch op and sets the child's tensor (`_direct_set_tensor`). No node, no
  edge, no name is created. Probe: a 784-site chain of `@` and `renormalize` took
  0.40 s the first time and 0.08–0.09 s on each later call (≈ 4.7×).
- Batch edges may change size between calls: a later call with batch 24 after 64
  hits the same successors (axes whose name contains `batch` are batch edges;
  `_compatible_shape`, `components.py:1305`, lets them differ when a tensor is set).
- `mul`/`div`/`add`/`sub` by a **tensor or number** key on `(node1,)` alone
  (`_check_first_div`, `operations.py:696`), so dividing by a freshly computed
  norm tensor still reuses the successor. (main: `_elementwise_key`;
  the replay of these, broken before, now passes the operand again; see below.)
- Successors survive until `reset`. Reuse needs **the same node objects** as
  operands: a node created anew each call (e.g. a fresh data node per call) makes
  every downstream operation a first call.

## forward, contract, trace, reset

- `TensorNetwork.forward` (`components.py:4999`): sets data nodes if missing,
  `add_data`, then **if there are no result nodes calls `contract()`, otherwise
  replays `_seq_ops`** (`self.operations[op](*args)` in order) and returns the last
  result. A replay does not run `contract()`'s Python, so any side effect there
  (bm4tc's log-norm accumulation in an overridden `_inline_contraction`) is skipped.
  That is why bm4tc used to `reset` around every forward. (main: such
  computations can be `tk.operation`s, and `contract` can return several nodes;
  see below.)
- Calling **`contract()` directly** instead runs its Python every time while every
  operation reuses its successor. bm4tc's forward does that now (no reset): values
  and gradients bit-identical, eager no-grad forward 2.5–3× faster.
- `trace(example)` (`components.py:4947`): `reset()`; one call with
  `_tracing = True`, which records in `_inverse_memory` how often each tensor
  address is read and whether it may be erased (only result and data nodes, and
  `stack_data_memory`); one normal call; `_traced = True`. From then on each
  `fn_next` calls `_check_inverse_memory`, which sets a tensor to `None` after its
  last read: intermediates are freed during the contraction.
- The trace bookkeeping can be had **without the replay**: set `_tracing = True`
  for the first `contract()` call, then `_traced = True`. With autograd this is
  safe (autograd keeps its own references). bm4tc: nothing held between calls, and
  the full-MNIST peak drops slightly (674 → 612 MiB). Without it, kept nodes hold
  one call's intermediates (645 MiB at 784 sites, r40, batch 64).
- `reset` (`components.py:4857`): resets `_traced`, `_seq_ops`, `_inverse_memory`;
  empties the successors of leaf and data nodes; gives leaf nodes their own tensors
  back (from stacks); **deletes every result node and every virtual node whose name
  contains `virtual_result`**. Data nodes stay. It is only needed when the structure
  of the contraction changes.
- `reset` **re-registers parameters** (`delattr(self, 'param_' + name)` then set
  again), which **reorders `module.parameters()`**. Code that zips parameters by
  position across a reset breaks (a bm4tc test did); optimizers and state dicts
  (objects, names) do not.
- `reset` is slow with many nodes: deleting renames the remaining same-named nodes
  (`_unassign_node_name`, `enum_repeated_names`), quadratic in the number of nodes
  with one name. 13k nodes: 0.9 s; 2.3 s of an 8.4 s sweep at 784 sites.
  `move_to_network` renames the same way (3.2 s of a 6.6 s sweep at 400 sites).

## Nodes, names, edges

- Names: a repeated name is enumerated (`name_0`, `name_1`, …) and renamed on every
  add/delete; this is the quadratic cost above.
- Kinds: leaf, data (`data=True`), virtual (`virtual=True`), result. `add_data`
  sets **every** data node in `_data_nodes` order (or `stack_data_memory` when all
  have one shape), so an extra data node breaks it: use virtual nodes for
  auxiliary inputs (bm4tc names them `virtual_result_sample_*`, so `reset` deletes
  them). `virtual_uniform` nodes (uniform MPS) are left alone by `reset`.
  (`contraction-primitives`: data nodes created directly also coexist in
  several sets; inputs per recorded contraction are issue #2.)
- Setting a tensor: `node.tensor = t` for non-result nodes that own their tensor;
  `set_tensor_from(other)` makes a node read another node's memory (`node_ref`),
  which is how a bra layer can share the ket's tensors without copying.
- Result nodes **inherit their operands' edges** (the same `Edge` objects).
  Connecting through an inherited edge connects the leaf's edge: call
  `reattach_edges(axes=[...])` on the result node first. A `permute` with the
  identity permutation is a cheap way to get a node of one's own to do edge
  surgery on.
- `@` (`contract_between`) refuses two nodes whose only shared edges are batch
  edges; `einsum` does that product. (`contraction-primitives`: they contract, as
  an outer product in the rest; see below.)
- `einsum` needs repeated **non-batch** subscripts to be connected edges.
- Duplicate axis names in one node are renamed `input_0`, `input_1`.
- Axis renaming on a node: `node.get_axis(name).name = new`.

## MPS (1.1.6 and develop)

- `contract(inline_input, inline_mats, renormalize, marginalize_output)`;
  `_inline_contraction` is a static helper a subclass can override (bm4tc does, to
  keep the norms). tk's `renormalize` operation divides by the norm and **discards**
  it. A `renormalize` that returns the log-norm as a node of its own would put the
  accumulation on tk's path (trace, successors, replay), and bm4tc's override and
  special forward could go (in bm4tc's plan as a deep option). (main:
  a `tk.operation` does that; `docs/custom_operations.md` §2.3 has the MPS.)
- 1.1.6 `contract` contracts every site with its data first, (B, r, r) per site,
  and keeps all of them: B·n·r² stored (153 MiB per sample at 784 sites, r160).
  Environment first (as develop's `norm` zips up) would store B·n·r·d.
- `copy()` in 1.1.6 gives fresh obc boundary vectors (develop shares them).
- bm4tc sets `auto_stack = False` (stacking would replace the leaf tensors per
  forward) and keeps a separate network (`norm_net`) for log Z.

## 1.1.6 → develop differences met so far

- `n_features` → `n_sites`; `oc` → `orth_center` (deprecated alias, warns).
- Nodes: develop has both `rank` and `ndim`.
- Inside `MPS`, develop's `super().set_data_nodes(...)` is `TensorNetwork`'s.
- develop's `tests/models/test_mps.py` in a torch 2.1 env: `test_to`,
  `test_to_contracted` (MPS, UMPS) and `test_deparameterize_preserves_boundary_runtime`
  fail on develop itself; `test_entropy` and `test_canonicalize_univocal_diff_dims`
  fail at random (unseeded shapes). Doctests need `tk` and `torch` in their globals.
  (main: both random failures root-caused and fixed; see below.)

## This branch (`sampling-efficiency`)

**Superseded for the sampler (2026-10-15)** by branch `sample-density` (the PR for
José, 4 commits on develop) and bm4tc's D106; see "Since: `sample-density`" below.
This branch keeps the D100–D103 history (step 3's open leg for the class, rejected
since: memory d_out^k for k output sites).

One commit per bm4tc step, each bm4tc `sampling.py` translated back to develop's
names (the translation of bm4tc's step 1 reproduces develop byte for byte):

1. `0a079fa` one domain per sample (`(n_samples, domain_size)` windows).
2. `9a08f59` each site's reduced density matrix first, candidates as quadratic
   forms; single-layer chains where variables are fixed; the bra layer built in
   place (no copy + move).
3. `a23003e` one layer (the ket) for a sweep (`in_condition`); rows with no usable
   weight drawn uniformly, one warning per call.
4. `e514b02` the sampling nodes kept between calls under a structure key; `reset`
   and `forward` remove them.

Not changed, candidates for separate changes: the quadratic renaming in `reset` and
`move_to_network`; `@` on batch-only edges; `einsum`'s connected-edge rule; edge
inheritance through result nodes; a norm-keeping `renormalize`; a public way to
"contract with reuse" (forward's replay forbids Python side effects in `contract`,
and the trace bookkeeping is only reachable through private flags);
`reset` reordering `parameters()`.

## Since: `sample-density` (bm4tc D106, 2026-10-15)

- **Double layers only where something is summed.** An environment is double-layer
  only if it contains a summed site (an output node; without `in_condition`, the
  inputs to the right of the drawn site); elsewhere the ket alone. With both
  environments single, a site's weights are |amplitude|² per candidate; otherwise
  quadratic forms on its density matrix.
- **Fixed sampling nodes.** Each step of the sweep (a right environment absorbing a
  site, the left one absorbing a drawn or summed site, a site's weights) is a small
  network of virtual leaves (`virtual_result_sample_*`, own edges connected once),
  built at its first use. Later sites set the leaves' tensors with
  `_direct_set_tensor` (no shape check; batch and candidate sizes may vary, since
  `_permute_contract_reshape` uses runtime shapes) and repeat the same operations,
  which hit their successors: no nodes per site, and memory does not grow with the
  number of sites except for the right environments kept for the sweep. Environments
  pass between steps as tensors, permuted to the leaf's axis order by name. Nothing
  is copied from the MPS and none of its edges is touched, so the `@`-on-batch-only,
  edge-inheritance and copy-and-move issues above are no longer met by `sample`.
- **Float32 precision (open).** A double layer is positive semi-definite in exact
  arithmetic only: it sums ket × bra products, so its rounding error grows like the
  square of the contraction's cancellation (one layer: linearly). develop has it
  too (−2.7e-3 of a row's maximum at 784 sites, bond 40, a narrow window);
  `sample-density` −8e-4 there, up to −4.6% with per-sample windows; float64 clean.
  Not solved; see bm4tc plan.md, "Sampling" step 6.

## Since: `main`, formerly `develop-unified` (2026-10-26)

The branch that unites develop's branches of the month: `zipup-input`,
`fix-cuda-device-tests`, `sample-density`, `sampling-efficiency` (notes and the tests of
repeated `sample` calls, which `sample-density` passes; its sampler is
superseded), and `custom-operations`. Then fixes found on the way.

**Operations core** (`docs/custom_operations.md` has primer, guide, design and
implementation):

- `tk.operation`: a function returning `(kernel, axes)` becomes an `Operation`.
  The function runs at the first call (it is `fn_first`'s planning), the kernel
  (any PyTorch) in every call and in the replay. Outputs are resultant nodes,
  with axes inherited from the operands (names and edges) or new. This removes
  the reason for the overridden `_inline_contraction` and the special forward:
  a renormalize that returns its log-norm is a 10-line custom operation.
- `contract` may return a tuple of nodes; `forward` returns their tensors, also
  in the replay and after `trace`, and the inverse memory never frees them.
- One core (`_build_operation`) builds the custom operations and `permute`,
  `tprod`, `mul`, `div`, `add`, `sub`, `renormalize`, `conj` (825 lines of
  hand-written `check_first`/`fn_first`/`fn_next` removed). Keys, names and
  public functions unchanged. These operations now work on StackNodes (they
  raised, except `renormalize`).

**Fixes:**

- Replay of `mul`/`div`/`add`/`sub` with a tensor or number: `_seq_ops`
  recorded `(node1,)`, so the second call of such a `contract` failed (missing
  `node2`). Now it records the operand.
- `tprod` in a traced network failed in every forward (`_check_inverse_memory`
  got the tuple of node refs).
- `reset` kept the successors of the virtual nodes it keeps (user auxiliary
  virtual nodes, `virtual_uniform`, `stack_data_memory`), pointing to deleted
  result nodes; a `contract` starting with such a node failed after `reset` and
  so after every `trace` (the TODO in `reset`).
- A new leaf node named as an operation raised `RecursionError` instead of the
  `ValueError`.
- `stack`/`unbind` indexed with lists of slices, which PyTorch reads as a
  tensor index in newer versions (it warned all over the suite); now tuples.
  pytest turns that warning into an error (`pyproject.toml`).
- `MPS.entropy` gave NaN when a squared Schmidt value underflowed in single
  precision (0 · log 0): `test_entropy`'s random failures (~15% of draws in
  complex64).
- `test_canonicalize_univocal_diff_dims` failed at random because float32
  rounding of some draws reaches 1e-3 (relative): it runs in float64 now, where
  the univocal form is exact to 1e-12.

Still open from the list above: the quadratic renaming in `reset` and
`move_to_network`; `@` on batch-only edges (done on `contraction-primitives`);
`einsum`'s connected-edge rule;
edge inheritance through result nodes; `reset` reordering `parameters()`; the
trace bookkeeping without the replay (private flags); float32 precision of
double layers in `sample`.

## Since: `contraction-primitives` (issue #1, 2026-10-28)

What a general tree contraction (one or two layers; observed, open or summed
variables) needed around tensorkrowch, items 1 and 2. Variants and measurements are in
`docs/custom_operations.md` (§2.9 guide, §3 design, §4.6 implementation).

- `contract_between(node1, node2, keep=None, rescale=False)` (and `@`,
  `contract_edges`), on the operations core:
  - `keep`: dangling axes that both nodes have are kept once, element-wise
    along them. Batch axes with the same name are always kept, by the same `bmm`
    batch dimension. Nodes sharing only batch or kept axes contract as an outer
    product in the rest.
  - `rescale=True`: the result is divided by its max |entry| per sample
    (detached; 1 if zero), and `(result, log_scale)` is returned.
  - The downstream `contract_rescale` (52 lines) becomes one call. Same values
    and gradients at rtol 1e-12.
- Rejected: a naming convention (dangling equal names kept), which breaks 710
  existing tests (MPS/MPO sites all have `input` legs); copy (δ) nodes, which
  need a neighbour lookup in the contraction, densify if the order touches δ,
  and cannot express the batch or a variable shared by three or more factors;
  a node-level einsum, which leaves the letter bookkeeping to the caller.
- Per-call inputs (the issue's item 3) moved to issue #2, programs per set
  of `contract` arguments with their own data nodes. Found here: data nodes
  created directly coexist in several sets, set with `node.tensor = t` (about
  3.7 µs against 0.2 µs for `_direct_set_tensor`) and freed by a traced
  replay; virtual nodes are never freed by the inverse memory.
- `contract_edges`' successors keep the key `(edges, node1, node2)` without
  `keep`/`rescale`; their hints are now the kernel. `a @ b` costs about +1 µs
  per call in lookup and replay (12.1 → 13.0 µs on small tensors).

## Since: `recorded-programs` (issue #2, 2026-10-28/30)

Several recorded contractions per network. Recipe, design and implementation
are in `docs/programs.md`.

- `forward(data, *args)` keeps one `Program` per set of normalised `contract`
  arguments and replays it from its second call (`fn_next` with the successors
  found once, no `check_first`). `tk.Program(net, fn)` records any other
  function. The lookup by arguments is private (`_program_for`).
- `max_programs` (default 1: a new set of arguments warns and resets, as
  before; `None` keeps all until `reset`). Programs are freed only together:
  evicting one would delete some nodes (name collisions or renames) and the
  `ParamStackNode` that `auto_stack` shares between programs.
- Each program has its own successor cache (`node._successor_caches`), so
  two contractions starting with the same operation on the same nodes no
  longer share (and lose) a step; and its own `_seq_ops` and inverse memory.
  The operations reach them through properties of the network, unchanged.
- Inputs: data nodes (one set per network), or input nodes that `contract`
  creates (`net.input_node(shape, axes_names)`), one set per program, for a
  different observed set per pattern. Input nodes are placeholders (zeros,
  batch 1): the first call records on them, fills them and replays.
- `free_intermediates`: one switch for every program (set by `trace`, kept by
  `reset`). Memory, not time: 64 MB → 0 held after a no-grad call (n=100,
  r=64, batch 256), same 27 ms.
- Speed: replay is as fast as per-pattern nodes re-run through the lookups
  (1.39 vs 1.42 ms); the gain is against resetting per switch (9.1 → 3.3 ms
  per training step with two contractions).
- Fixes on the way: the inverse memory is keyed by the node storing the
  memory, not its name, so freeing survives a rename of nodes (`x` → `x_0`),
  which raised `KeyError`; eager operations no longer hit (and count against)
  a program's cache; `conj` of a real parameter lost its gradient on the first
  call (the resultant node detached the parameter that `conj` returned).
- `MPS.sample` keeps `_sample_kernel`: its steps can be programs, but not in
  fewer lines, for about 13 of 70 µs per step.
- After the review of #4: `copy_node(node)` (a program's copy sharing a
  node's tensor, new edges, deleted by `reset`) replaces hand-made
  `virtual_result_*` copies; a program with input nodes called without data
  raises; pickling drops successors and recordings (`torch.save` failed on
  local kernels, on `contraction-primitives` too); the internal flags are
  `_recording` and `_freeing` (were `_tracing`, `_traced`). The notes above
  that name `_tracing`/`_traced` describe the code before this branch.
