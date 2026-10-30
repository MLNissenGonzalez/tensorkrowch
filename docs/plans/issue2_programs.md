# Issue #2: several recorded contractions per network (plan)

Branch `recorded-programs` (from `contraction-primitives` with `main` merged). Written
2026-10-28 before any code for #2; it is the hand-over to the next session.

To resume: read this file, `docs/custom_operations.md` (the operation machinery it builds on)
and `TK_NOTES.md`, then fork issue #2. Item 3 below asks one question of the maintainer
before code.

## 1. Critique of the downstream forward (tpm-tn-benchmark, `main` bc5b5d8)

`tpm_tn/core/network.py`: `TreeNetwork.contract(pattern)` builds a `_Plan` per pattern and
runs it; `forward` raises `NotImplementedError`. Read only; nothing here was changed there.

1. **The shared-cache failure (section 2) does not occur there.** Each `_Plan` (key: layout,
   `frozenset(observed)`, `open`, `keep_off_diagonal`) owns its virtual nodes, one per tree
   node per layer (`virtual_result_plan{k}_ket3`, ...) plus the φ / Σφ / Gram inputs, wired
   once. Operations never take the network's own ParamNodes as operands, so no two plans
   share an operand node and so no successor-cache entry.
2. **But nothing is replayed.** `_Plan.run` is a Python loop over its steps on every call; each
   `contract_rescale` goes through the operation's cache lookup (`check_first`, key build,
   dict lookup) before its kernel. No trace, no inverse memory.
3. **Private tensorkrowch API on every call** (`_Plan.run`, network.py:378-397):
   - `_direct_set_tensor` refills every leaf each call: n ket and n bra copies of the
     parameters, and the Σφ / Gram inputs, which depend only on the feature map and so are
     constant;
   - `.to(dtype=..., device=...)` per leaf per call;
   - `_save_in_network(None)` to free operands by hand (what tk's inverse memory does).
4. **Unbounded growth.** Measured on a 12-variable chain (rank 6, float64) after one
   likelihood, one norm and the 11 conditionals of a sampling sweep: 13 plans, 1022 resultant
   nodes, 511 virtual nodes, and tk's `_seq_ops` at 511 entries that nothing ever replays.
   Nothing is freed except by `reset`, which is quadratic in the node count (TK_NOTES: 13k
   nodes, 0.9 s). Sampling asks one pattern per variable per order, so the count grows
   without bound over a run with varying orders.
5. **#39's baseline (d)** ("a copy of the network per pattern") needs
   `copy(share_tensors=True)`. tk's `MPS.copy` has it; the base `TensorNetwork.copy` does
   not, so a generic subclass like `TreeNetwork` cannot use it as is.
6. **Stale branch name.** `environment.yaml` installs
   `git+https://github.com/MLNissenGonzalez/tensorkrowch@develop-unified`, and `GUIDE.md`
   (3 places), `tests/integration/test_network_tk_parity.py` and #39 name it. That branch is
   now `main` and the old name is deleted: a fresh `conda env create` fails until they say
   `@main`.
7. `log_abs_phase` becomes two lines of torch once `forward` returns several outputs
   (on `main`): the log-scales are ordinary outputs then.

**Correction (2026-10-30, measured after implementing #2).** Points 2 and 3 are
not a speed problem. Keeping one's own nodes per pattern and re-running the
operations through their lookups is as fast as replaying a program (1.42 vs
1.39 ms per forward with grad, 1.00 vs 1.01 ms without, on a 12-variable chain;
`docs/programs.md` §2.4): kernels and per-operation bookkeeping dominate, not the
lookup. The downstream design was sound in substance (its per-plan nodes are the
per-program caches, by hand). What remains of the critique: it uses private tk
API, frees by hand, and is specific to `TreeNetwork`; and point 4's growth is
the same with programs that are never freed (`max_programs=None`).

Why this was not surfaced earlier: the downstream repo was first read in this session; the
operation work before it looked only at tensorkrowch.

## 2. The problem in tensorkrowch

- `forward(data, *args)` records one contraction (`contract(*args)`) and replays its
  `_seq_ops`; different `args` warn and `reset` (pinned by `test_trace`).
- An operation's cache lives on its first operand (`_successors[name][key]`), and `_seq_ops`
  is appended only in `fn_first`. Two recordings that share an operand node (e.g. two
  `MPS.contract` variants over the same ParamNodes) hit the same cache entry: the second
  goes through `fn_next`, so its own sequence misses that step and replays incompletely.
- The edge model (one edge per axis) means programs needing different connectivity need
  their own nodes sharing tensors (`set_tensor_from` / virtual nodes), as the downstream
  plans do by hand.

## 3. Plan

1. **One `Program` class** holding: its recorded steps as (operation, args, stored
   successor), so replay calls `fn_next` with the successor directly, no lookup; its own
   inverse memory, outputs (`_contract_outputs`), traced flag, and key (normalised contract
   arguments); the nodes it created (resultant and input) and its data nodes.
2. **On the network**: a dict of programs keyed by normalised arguments;
   `net.program(*args, fn=None)` returns the program or records it (`contract` by default,
   or another function, e.g. a sampler step); `forward(data, *args)` =
   `self.program(*args)(data)`. One mechanism; today's forward is the one-key case.
3. **Opt-in, question for the maintainer**: `max_programs=1` by default keeps today's
   warn-and-reset (and `test_trace`); `n` evicts least recently used (deleting that
   program's nodes and cache entries); `None` keeps all until `reset`.
4. **Per-program caches** for programs sharing operand nodes: one more dict level
   (program) in the successor cache, consulted only while recording.
5. **The six hand-written node operations** (`split`, `split_edge`, `merge_edges`, `stack`,
   `unbind`, `einsum`) go through the active program's state, or move onto the core
   (`_build_operation`), whichever is less code.
6. **Inputs**: `program(data)` sets its data nodes; constant leaves are set once; ket copies
   share memory via `set_tensor_from`; the bra's conjugate is a recorded `conj`.
7. **Tests** (float64, real and complex): two argument sets alternating give no warning and
   no reset (with the opt-in); values and gradients equal a fresh contraction; eviction frees
   that program's nodes only; the existing suite is unchanged.
8. **`MPS.sample` on programs, only if it comes out shorter** (lines counted). It already
   uses a small number of step networks, constant in the number of variables (keyed by
   structure: right, left, site steps), so the gain is not in nodes: it is a hand-made
   program cache (`_sample_kernel` + `kernels` dict) that fills inputs with private
   `_direct_set_tensor`, runs each step's Python and the operation lookup every call, and
   frees nothing. Programs would replace that plumbing; expect a modest gain.
9. **Downstream sketch for #39** (in the PR text, not in that repo): the body of
   `TreeNetwork.contract(pattern)` (wiring + path search) becomes the recording function;
   `forward(φ inputs, pattern)` replays it; `_Plan.run` (21 lines), `_fill` and the private
   calls go away. Count lines against them.
10. **Docs**: a section in `docs/custom_operations.md` (or its own page) and `TK_NOTES.md`; the
    PR description lists the variants tried and why this one.
