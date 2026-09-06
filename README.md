# ncpu-computer

`ncpu-computer` is a research project for learning computation inside a small
neural cellular automaton (NCA). A shared local neural rule evolves a spatial
state over time. Inputs and outputs are strings written on a one-dimensional
logical tape embedded in a two-dimensional grid.

The immediate objective is to learn individual string transformations such as
binary addition, reversal, bitwise NOT, and parity. The medium-term objective
is to place a task-specific program in the initial state so that one fixed
local rule can perform different computations when given different programs.

The repository contains the complete fixed-geometry, single-task baseline:
ternary codecs, tape layouts, task datasets, the local NCA rule, long-window
training, resumable checkpoints, exhaustive or sampled evaluation, inference
interpreters, validation, tests, and a runnable notebook. Program-conditioned
and mixed-geometry training remain deliberately deferred research stages.

## Run it

From this directory on Windows:

```powershell
conda env create -f environment.yml
conda activate slackenv
pip install -e ".[dev,notebook]"
pytest -q
jupyter lab run/run.ipynb
```

The environment file also installs this package in editable mode. Installing
the development extra adds the pinned formatter and linter used by the test
workflow.

A minimal training setup is:

```python
from ncpu_computer import (
    ExperimentConfig,
    TaskDataset,
    Trainer,
    addition_task,
    validate_experiment,
)

config = ExperimentConfig()
dataset = TaskDataset.from_task(
    addition_task(4), config.geometry.tape_slots
)
print(validate_experiment(config, dataset))

trainer = Trainer(config, dataset)
trainer.fit("checkpoints/seed_0")
```

The addition notebook exposes every task, geometry, model, training,
evaluation, and GIF setting in its first code cell. That cell validates the
experiment and prints the physical tape layout before any optional action.
Training is not launched automatically: set its explicit `RUN_TRAINING`
switch when ready. The notebook default batch size is 64 so its 200-step
backpropagation graph fits a 4 GiB GPU.

For unary binary-string experiments, open
`run/simple_binary_tasks.ipynb`. Its `TASK_NAME` switch selects either reversal
or bitwise NOT. It trains on every string up to a chosen length and evaluates
on strings of one exact longer length, preserving leading zeroes throughout.

## Repository structure

```text
src/ncpu_computer/
  config.py       explicit geometry, model, and training configuration
  tape.py         ternary tensor codec, strided layout, inference interpreters
  tasks.py        generic string tasks, built-in tasks, datasets and masks
  model.py        perception and shared residual NCA update rule
  training.py     objectives, optimization, checkpoints, multi-seed runs
  evaluation.py   tensorized metrics and single-example inference
  validation.py   fast checks of the experiment's core invariants
  visualize.py    role-aware annotated GIF rendering of NCA trajectories
run/run.ipynb     binary-addition training and evaluation workflow
run/simple_binary_tasks.ipynb
                  reversal/bitwise-NOT and length-extrapolation workflow
tests/            focused CPU regression tests
```

## Central idea

The learned local rule is the computational mechanism. The grid supplies
working memory, and repeated NCA updates supply computation time. The same rule
is applied at every cell and at every timestep:

```text
input string -> initial spatial state -> repeated local updates -> output string
```

The rule contains no hand-written arithmetic, routing, carry propagation,
string-length logic, or position-specific parameters. Only the external
encoder, decoder, and training objective define how data communicates with the
substrate. Internal computation is allowed to emerge from end-to-end
supervision.

Because the local rule is independent of grid size, a trained model can be run
on a longer tape without increasing its parameter count. Fit on the training
length, extrapolation to longer strings, stability over time, and reuse across
tasks are separate scientific questions and must be measured separately.

## Logical alphabet

The interface uses ternary strings over the symbols `0`, `1`, and `B`:

| Logical symbol | Tensor target | Meaning |
|---|---:|---|
| `0` | `-1` | Binary zero |
| `B` | `0` | Blank or separator |
| `1` | `+1` | Binary one |

`B` normally provides blank space and separates values. It remains a genuine
symbol, however, and future tasks may assign it other learned roles.

The core model operates on tensors and does not assume that a string represents
an integer. Integer codecs are optional outer utilities. When an integer is
encoded, its minimal binary representation is used: leading zeroes are omitted,
and integer zero is represented by the one-symbol string `0`.

For example, the two operands `111` and `100` are serialized as:

```text
logical:  1  1  1  B  1  0  0
tensor:  +1 +1 +1  0 +1 -1 -1
```

## Tape geometry

One horizontal row of a two-dimensional NCA grid contains the logical tape.
Logical symbols occupy regularly spaced cells with configurable horizontal
stride `s`; the default is `s = 2`. Physical cells between logical positions
belong to the computational medium but are not tape symbols.

The grid has independently configurable empty space above, below, left, and
right of the tape. The active logical tape excludes the left and right spatial
borders:

```text
upper empty space

left border | x0 . x1 . x2 . ... . xL | right border

lower empty space
```

Here `x0 ... xL` are logical tape cells and `.` denotes physical cells between
them when the stride is greater than one. Input is left-aligned at `x0`.

Input and output use the same logical positions. The NCA must overwrite the
input with the output, which may be shorter or longer. A particular execution
has the finite capacity supplied by its grid, but the rule itself has no
fixed-string-length parameter and can be evaluated on wider grids.

Only the strided logical tape cells participate in tape readout and tape loss.
Physical gap cells, borders, and other grid locations remain latent computation
space.

## State channels

The default state has five channels:

| Channel | Initial role | Mutability |
|---:|---|---|
| 0 | Program | Read-only, initially zero everywhere |
| 1 | Input/output tape | Always mutable |
| 2 | Computation | Mutable |
| 3 | Computation | Mutable |
| 4 | Computation | Mutable |

The program channel is exactly zero at initialization and is restored unchanged
after every update while the model is trained on one task at a time. Reserving
it now keeps the state interface compatible with the later program-conditioned
model. The I/O channel is always mutable because it is used for both reading
and writing.

## Local neural rule

The initial architecture follows the small, validated design explored in
`ncpu-simplified`:

1. Apply a bank of 3x3 depthwise perception kernels to every state channel.
2. Mix all perceived features with a shared 1x1 convolution.
3. Apply ReLU.
4. Produce a per-channel update with a shared 1x1 convolution.
5. Optionally modulate the update with a learned gate.
6. Apply an optional stochastic per-cell fire mask.
7. Add the update residually to the previous state.
8. Optionally clip the mutable state.
9. Restore read-only channels exactly.

Perception may combine fixed identity, Sobel-X, Sobel-Y, and Laplacian filters
with configurable learned 3x3 kernels. Padding mode, gates, fire rate, clipping,
channel count, kernel bank, and hidden width remain configurable.

The default hidden width is `96`. With five channels, four perception kernels,
one shared learnable 3x3 kernel, and no gate, the corresponding rule has 2,505
trainable scalar parameters. This is a starting point rather than a claim that
larger rules are inherently better. Architecture comparisons must use matched
seeds and training conditions.

The delta-producing projection should begin at zero so that the initial model
implements identity dynamics. This avoids imposing arbitrary destructive
dynamics before learning begins.

## Training semantics

Each example supplies:

- an input ternary string;
- a target ternary string;
- a logical tape capacity;
- task metadata used only by the external data codec.

Both strings are embedded into complete tape tensors. The input occupies the
I/O channel at timestep zero. The target begins at the same leftmost logical
position and is padded with `B` through the final logical tape position.

Targets affect only loss calculation. They are never injected into the evolving
state, never used to overwrite predictions, and never exposed to the NCA.

The NCA first evolves for a configurable number of free steps. Loss is then
applied across a long supervision window. If there are `F` free steps and `S`
supervised steps, the supervised states are:

```text
F + 1, F + 2, ..., F + S
```

This asks the model both to compute an answer and to retain it. It does not imply
stability outside the supervised interval.

### Base loss

The base objective is ordinary mean squared error over:

- the batch;
- every supervised timestep;
- every logical tape position.

The target values are exactly `-1`, `0`, and `+1`. Therefore zero base loss
means exact numerical reproduction of the full target tape, including blank
padding. Blank-heavy tapes intentionally contribute proportionally more blank
terms; the base loss is not class-balanced.

No base loss is applied to non-tape cells or to channels other than I/O.

### Optional structural losses

Two independently averaged auxiliary terms may reweight structural blank
regions:

```text
loss = base_mse
     + terminator_weight * terminator_mse
     + tail_weight * tail_mse
```

Both weights default to `0`.

In single-output mode, `terminator_mse` acts on the first target `B` following
the output. In multiple-output mode, single `B` symbols between values are
ordinary separators covered by the base loss, while `terminator_mse` acts on
the final `BB`. `tail_mse` acts on every logical position following the
terminator. These terms add weight; they do not remove the same cells from the
base MSE.

## Tensor readout and quantization

Training and batched evaluation remain tensorized. Logical positions are
gathered from the I/O channel into a tensor with shape:

```text
(batch, tape_length)
```

For a temporal rollout, an additional time dimension is retained. There is no
conversion to Python strings inside the model or training path.

Discrete readout uses the literal threshold `0.333`:

```text
value >  0.333  -> 1
value < -0.333  -> 0
otherwise       -> B
```

The exact boundary values `-0.333` and `+0.333` decode as `B`. The decimal
literal is intentional and must not be replaced by an exact or computed `1/3`.

MSE and discrete correctness measure different properties. A value can have the
correct discrete symbol while remaining far from its numerical target.

## Inference interpreters

The complete logical tape is always read and quantized in parallel first. For a
single interactive inference, this tensor may then be rendered as a string such
as:

```text
101B11BB0
```

Interpretation is an outer inference-only operation. It is not part of the NCA,
training loss, or raw tape readout.

Two inference interpreters are provided:

### Single output (default)

Read from the left and stop completely at the first `B`:

```text
1011BBB0 -> binary string 1011 -> integer 11
```

### Multiple outputs

A single `B` separates values. Two consecutive blanks `BB` terminate the
complete output, and all subsequent symbols are ignored:

```text
101B11BB0 -> binary strings [101, 11] -> integers [5, 3]
```

Task-specific interpreters may be added later without changing the raw tensor
interface. A terminated nonminimal binary string such as `01` remains a valid
string output, but the integer view is deliberately unavailable because integer
codecs require minimal binary representation.

## Evaluation

Scientific evaluation should report at least:

- mean full-tape MSE across the chosen window;
- ternary symbol accuracy;
- whole raw-tape exact accuracy;
- interpreted task-output exact accuracy;
- accuracy stable across the complete supervision window;
- per-timestep curves and best timestep;
- extrapolation to longer tapes than those used in training;
- behavior beyond the trained time window;
- variation across independently trained seeds.

An interpreted answer is not enough by itself: raw-tape accuracy exposes extra
symbols, missing blanks, and other representations that an outer interpreter
might ignore.

## Program-conditioned computation

The medium-term model will generate the initial program channel from a task
index:

```text
task index -> task-specific spatial program -> shared NCA rule -> task behavior
```

Candidate mechanisms include a directly learned task embedding and a
coordinate-conditioned CPPN that can generate a program over variable grid
sizes. The program is a pattern in the latent state, not merely the existence of
a channel.

The intended progression is:

1. Train and validate one task at a time with a zero program.
2. Train one shared rule with a separate learned program for each task.
3. Freeze the rule and learn only a new program for an unseen task.
4. Study variable-size programs, grids, and computation times.

Only the third stage directly tests whether the learned rule behaves as reusable
computational hardware rather than storing every task in its weights.

## Variable geometry and time

The current implementation uses a simple fixed grid and fixed rollout
schedule. A later training regime may mix examples with different active grid
sizes inside padded tensors. An external validity mask would then define the
real grid for each example, force the exterior to zero after every update, and
exclude it from loss and readout.

Sampling active widths and computation times is intended to reduce dependence
on one boundary distance or one exact temporal schedule. This is deferred until
the fixed-size implementation is correct and scientifically validated.

## Engineering principles

- Prefer the simplest complete design.
- Keep the encoder, NCA dynamics, tensor readout, and inference interpreter
  separate.
- Enforce invariants directly rather than adding fallbacks or repair paths.
- Fix causes, not symptoms; do not patch around incorrect logic.
- Keep configuration explicit and checkpointed.
- Isolate data sampling RNG from initialization and stochastic evolution where
  required for fair comparisons.
- Resume only from exactly compatible configurations.
- Preserve optimizer and RNG state in resumable checkpoints.
- Select checkpoints using exhaustive or clearly defined validation, not
  training loss alone.
- Compare architectures with matched seeds, data, schedules, and evaluation
  windows.
- Do not claim universality from finite-length extrapolation.
- Do not launch expensive training as a substitute for focused correctness
  tests.

## Research lineage

This project develops from the local neural-computation direction explored in
[`ncpu-simplified`](../ncpu-simplified) and is primarily inspired by Iliya
Zhechev's [`ichko/ncpu`](https://github.com/ichko/ncpu). Its program-state and
shared-substrate direction is informed by *Emergent Models: Intelligence from
Tiny Substrates* and the distinction between learned hard parameters and a
task-selecting latent program.

The present system is not claimed to be universal. Its purpose is to test, with
minimal and inspectable models, which ingredients lead from single-task local
computation toward a reusable programmable substrate.
