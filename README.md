# ncpu-computer

**ncpu-computer** tests whether a small local neural cellular automaton (NCA)
can learn algorithms that remain correct when the tape and computation time
change. The current design deliberately returns to the simplest interface that
worked best in earlier experiments: one tape and one shared input/output
channel.

Bit-NOT, a local operation, learned quickly in earlier versions. Reversal could
fit trained lengths but did not reliably extrapolate, while addition did not
learn reliably. This version targets that failure directly by varying tape
capacity, input length, and rollout time during training, without changing the
local rule.

## A type-independent interface

The NCA does not receive integers or task-specific structures. It reads and
writes strings over one alphabet:

| Symbol | Cell value | Role |
|---|---:|---|
| 0 | -1 | binary zero |
| B | 0 | blank or separator |
| 1 | +1 | binary one |

Any finite data type can be serialized externally into a {0, 1, B} string. The
learned system therefore operates through one general interface, independent
of the source data type. B fills unused capacity and may separate values. There
is no learned encoder or decoder.

Strings are left-aligned. Training examples are ordered by length, beginning
with the completely blank input:

~~~text
BBBB...
0BBB...
1BBB...
00BB...
01BB...
10BB...
11BB...
~~~

The complete target tape is supervised. If the target is 101 on a ten-slot
tape, the target tensor represents 101BBBBBBB, and all seven blank cells
contribute to the ordinary mean-squared error.

At inference, continuous tape values are converted to raw symbols with the
literal thresholds:

~~~text
value >  +0.333  -> 1
value <  -0.333  -> 0
otherwise        -> B
~~~

An external interpreter may then stop at the first blank, split on blanks, or
apply another data-specific convention. This interpretation is not part of
training.

## Layout

A tape has N logical cells at configurable stride S. It lies on the central row
of a grid with symmetric vertical space and equal horizontal space on both
sides:

~~~text
height = 2 * vertical_space + 1
width  = 2 * horizontal_space + N * stride

. . . . . . . . . . . .
. . . T . T . T . T . .
. . . . . . . . . . . .
~~~

For the default stride 2, each T is the right-middle cell of its two-column
stride block. Only these logical cells are injected and supervised; all other
cells are working space. Horizontal perception always uses zero padding.
Vertical perception can use either zero padding or circular wrapping.
Horizontal wrapping is never used.

The NCA state is:

~~~text
program channels | shared mutable I/O channel | computation channels
~~~

The input is written into the I/O channel. That same channel evolves and is
read as the output.

## Repeated task program

A task program is a small tile with shape:

~~~text
program_channels x grid_height x stride
~~~

It starts at absolute horizontal coordinate zero and repeats across the full
grid, including the boundary space:

~~~text
program[:, y, x] = tile[:, y, x % stride]
~~~

Tiles are adjacent and never overlap. Their parameter count is independent of
tape capacity, and every tape cell sees the same program phase.

Three modes are available:

- **zero**: fixed all-zero program; the default single-task baseline.
- **learned_read_only**: a learned initial tile fixed during each rollout.
- **learned_mutable**: a learned initial tile that may also evolve.

The shared update rule and program can be optimized jointly or separately.
They have independent weight decay. A future task can therefore adapt only a
small program while retaining a frozen rule.

## Local computation

Every cell applies the same update at every timestep:

~~~text
perception = depthwise_3x3(state)
delta      = pointwise_network(perception)
state      = state + mutable_mask * delta
~~~

The pointwise network is 1x1 -> ReLU -> 1x1. Its final projection starts at
zero, so the initial dynamics are exactly the identity. Perception kernels,
hidden width, gating, stochastic firing, and state clipping remain
configurable.

Optional Gaussian noise is added after perception during training only. Its
standard deviation is linearly annealed across optimizer updates. The default
is zero throughout, so no noise tensor is generated.

## Variable-shape training

One optimizer update contains N_TRIALS sequential trials. Each trial has its
own tape capacity, maximum input length, and computation time. Values are
sampled from coupled non-overlapping strata, so a default three-trial update
contains distinct small, medium, and large cases.

For each trial:

1. Build every binary input from length zero through the sampled maximum.
2. Sample the same batch size for every enabled task.
3. Run the trial on its own unpadded grid.
4. Average MSE over tasks, examples, supervised times, and all tape cells.
5. Backpropagate trial_loss / N_TRIALS immediately.

After all trials, gradients are clipped once and the optimizer steps once.
Different shapes are never padded or masked into one batch, and only the
largest individual trial graph needs to be resident at a time.

The free-evolution time is resampled on every update. The supervision-window
length is round(supervision_ratio * free_steps); the default ratio is 1.6. The
complete configured ranges are active from the first update.

## Tasks and evaluation

Built-in string tasks are:

~~~text
copy       101 -> 101
bit_not    101 -> 010
reverse    101 -> 101
parity     101 -> 0
append_0   101 -> 1010
append_1   101 -> 1011
~~~

The notebook defaults to one task, while the library supports balanced
multi-task batches, one tile per task, and one shared rule.

Evaluation uses explicit deterministic cases. Each case specifies tape
capacity, exact input length, free steps, and supervised steps. Reports include
MSE, symbol accuracy, exact full-tape accuracy, interpreted semantic accuracy,
stability across the supervision window, and best-timestep statistics.

## Running experiments

On Windows:

~~~powershell
conda env create -f environment.yml
conda activate slackenv
pip install -e ".[dev,notebook]"
pytest -q
jupyter lab run/run.ipynb
~~~

[run/run.ipynb](run/run.ipynb) is the only experiment notebook. Its cells are
organized as follows:

1. all scientific settings, structural validation, layouts, channel roles, and
   example ordering;
2. training, loading, and checkpoint controls;
3. post-training evaluation controls;
4. visualization controls.

The GIF shows only the evolving shared I/O channel on a fixed [-1, +1] scale,
with the selected task's initial program tile displayed separately.

Checkpoint format 5 stores the full configuration, ordered task signatures,
model, program, optimizer groups, progress, history, and all random-generator states.
Resuming reproduces trial sampling, batches, firing masks, and perception
noise. Older formats are intentionally rejected.

This is an experimental study of programmable local computation; it does not
claim computational universality.
