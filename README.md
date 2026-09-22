# ncpu-computer

`ncpu-computer` studies whether one small local neural cellular automaton (NCA)
rule can learn several computations. The task is selected by a learned program,
the grid supplies working memory, and repeated local updates supply computation
time.

The programmed model follows three observations from the earlier single-task
direct-tape experiments: bit-NOT, a strictly local operation, learned quickly;
reversal learned within the training lengths but did not extrapolate to longer
strings; addition neither learned reliably nor generalized. Separating a
learned task program from one shared rule makes the next question explicit:
which reusable local dynamics can support computations with different spatial
requirements?

## A type-independent tape interface

The model does not receive Python integers or task-specific tensors. It reads
and writes strings over one fixed alphabet:

| Symbol | State value | Meaning |
|---|---:|---|
| `0` | `-1` | binary zero |
| `B` | `0` | blank or separator |
| `1` | `+1` | binary one |

`B` fills unused tape capacity and can separate fields in structured strings.
An external deterministic codec may serialize integers, Boolean sequences,
tuples, or other finite data into `{0, 1, B}`. The NCA therefore has one
general interface independent of the original data type. There is no learned
input encoder or output decoder.

Strings are left-aligned and shorter strings are blank-padded:

```text
BBBB   empty
0BBB
1BBB
00BB
01BB
```

The full output tape is read in parallel. Continuous values are converted back
to symbols using the literal threshold `0.333`:

```text
value >  +0.333 -> 1
value <  -0.333 -> 0
otherwise       -> B
```

Interpretation of that raw ternary string is an external inference step.

## Geometry

The logical tape has `N` positions on one grid row. Adjacent logical positions
have configurable stride `s`; intervening cells and the four borders are NCA
working space initialized to zero.

```text
height = top + 1 + bottom
width  = left + (N - 1)s + 1 + right

working space above
left | x0 . x1 . x2 . ... . x(N-1) | right
working space below
```

Only `x0 ... x(N-1)` are supervised tape positions. A target shorter than `N`
is completed with `B`, so every unused slot is explicitly trained toward zero.

Full-grid programs depend on this exact geometry. Training and evaluation must
therefore use the same tape capacity, stride, and borders. Length extrapolation
is tested inside that fixed capacity by training on short strings and testing
on longer strings.

## Programs and channels

Each task has its own learned initial program, selected by task index. All tasks
share the same perception and update-rule parameters.

Two program placements are supported:

- `grid`: a learned value at every grid cell and program channel;
- `tape`: learned values only at logical tape positions, with exact zero
  elsewhere.

Programs start as small independent zero-mean random values. They are read-only
during evolution by default, but remain differentiable and are trained through
their effect on the computation. `program_mutable=True` allows the local rule
to update them. Program parameters use their own optimizer weight decay.

The default separate-I/O state is:

```text
program | input | output | computation channels
```

The input may be mutable or frozen. The output starts at zero and is always
mutable. Optional shared-I/O mode instead uses:

```text
program | shared input/output | computation channels
```

The shared channel starts with the input, evolves, and is read as output; it
cannot be frozen.

## Local computation

At every cell and timestep, the NCA applies the same rule:

```text
D_t     = U_theta(P(X_t))
Y_t     = clip(X_t + M * D_t)
X_(t+1) = M * Y_t + (1 - M) * X_t
```

`P` is a configurable bank of depthwise `3 x 3` perception kernels.
`U_theta` is a shared two-layer `1 x 1` network with a ReLU hidden layer, and
`M` marks mutable channels. The output projection begins at zero, so initial
dynamics are exactly the identity. Gating, stochastic firing, clipping,
padding, and learned perception kernels remain configurable.

The rule's parameter count is independent of tape length and task count. The
program bank grows with the number of tasks and, in `grid` mode, with grid size.

## Tasks

The notebook trains these tasks together:

```text
copy       101 -> 101
bit_not    101 -> 010
reverse    101 -> 101
parity     101 -> 0       (odd number of ones -> 1, even -> 0)
append_0   101 -> 1010
append_1   101 -> 1011
```

Every task receives the same ordered inputs: the empty string, then every
binary string of lengths `1 ... TRAIN_MAX_LENGTH`. Empty-input behavior is:

```text
copy, bit_not, reverse: BBBB -> BBBB
parity, append_0:       BBBB -> 0BBB
append_1:               BBBB -> 1BBB
```

Tape capacity is chosen explicitly and is never enlarged automatically. The
dataset construction fails if any input or target does not fit; append tasks
therefore require at least `TRAIN_MAX_LENGTH + 1` slots.

## Balanced multi-task training

Each optimizer update samples the same number of examples from every task,
concatenates and shuffles them, selects the corresponding programs, and runs
one tensorized rollout. For each task, the loss is ordinary MSE over examples,
all supervised timesteps, and the complete logical tape. The task means are
then averaged equally:

```text
task_loss = mean((output_tape - target_tape)^2)
loss      = mean(task_loss over tasks)
```

This is equivalent to averaging the task gradients, without six sequential
rollouts. Targets are used only in the loss and are never injected into the
evolving state.

After `F` free steps, supervision is applied over a window of `S` states:

```text
F + 1, F + 2, ..., F + S
```

Evaluation reports each task separately and an equally weighted aggregate.
Inference always requires an explicit task name.

## Running experiments

On Windows:

```powershell
conda env create -f environment.yml
conda activate slackenv
pip install -e ".[dev,notebook]"
pytest -q
jupyter lab run/run.ipynb
```

[`run/run.ipynb`](run/run.ipynb) works from either the repository root or the
`run` directory. Its first code cell exposes the task list, fixed geometry,
program and channel modes, local rule, optimizer, supervision, and validation
settings. Structural checks and the layout print unconditionally. Training,
post-training evaluation, and GIF generation have controls in their own cells.
The GIF displays only the channel used for readout on a fixed `-1 ... +1`
color scale.

Checkpoint format 4 stores the ordered tasks, dataset signatures, program bank,
fixed geometry, I/O mode, model, optimizer, and random states. Older formats
are rejected rather than adapted.

This project is an experimental test of programmable local computation; it
does not claim computational universality.
