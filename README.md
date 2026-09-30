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
width  = 2 * horizontal_space + (N - 1) * stride + 1

. . . . . . . . . . .
. . T . T . T . T . .
. . . . . . . . . . .
~~~

More precisely, logical cell `i` is at `horizontal_space + i * stride`.

The first and last tape cells therefore have exactly `horizontal_space`
physical columns outside them. Only logical cells are injected and supervised;
all other cells are working space. Horizontal perception always uses zero padding.
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

Its configurable origin is either absolute horizontal coordinate zero or one.
With the default origin `1`, column zero is initialized to zero and complete
tiles repeat from column one:

~~~text
0 | tile | tile | tile | ...
~~~

Origin `0` instead repeats from the first grid column; because the symmetric
grid has odd width at the default stride, the last repetition can be partial.
Tiles are adjacent and never overlap. Their parameter count is independent of
tape capacity. In mutable mode, the leading zero column is only an initial
condition and may evolve normally.

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
convolution_features = depthwise_3x3(state)       # optional
attention_features   = local_attention(state)     # optional
perception            = concatenate(enabled features)
delta                 = pointwise_network(perception)
state                 = state + mutable_mask * delta
~~~

Convolution and attention can be enabled independently, but at least one must
be active. Local attention uses the complete cell state, a configurable square
radius, and one or more heads. Its default radius is one, giving a 3x3
neighborhood. Non-wrapping positions outside the grid are masked before the
softmax. Vertical wrapping uses unique periodic neighbors even when the radius
exceeds the three-row grid height; horizontal wrapping is never used.

Attention has no positional encoding by default. An optional learned radial
bias distinguishes Chebyshev-distance rings without encoding direction or
absolute position. No state normalization is applied. An optional smooth norm
cap can limit queries and keys independently in each head, while values remain
uncapped so their magnitude continues to carry information.

The pointwise network is 1x1 -> ReLU -> 1x1. Its final projection starts at
zero, so the initial dynamics are exactly the identity. Perception kernels,
attention size, hidden width, gating, stochastic firing, and state clipping
remain configurable. When gating is enabled, the same pointwise network
produces both the delta and its gate after convolution and attention have been
combined.

Optional Gaussian noise is added after perception during training only. Its
standard deviation is linearly annealed across optimizer updates. The default
is zero throughout, so no noise tensor is generated.

## Variable-shape training

One optimizer update contains one sequential trial for each paired base case.
The defaults are:

~~~text
tape slots        5  7  8  9
max input length  3  5  6  7
~~~

At every update, tape capacity and maximum input length receive independent
uniform percentage perturbations (±30% by default), each rounded with
`floor(x + 0.5)`. A sampled pair is accepted only when every active task leaves
at least one blank after both its input and output. If it does not, the tape
capacity is retained and only the input-length draw is repeated. Trial order is
then shuffled.

For each trial:

1. Build every binary input from length zero through the sampled maximum.
2. Sample the same batch size for every enabled task.
3. Run the trial on its own unpadded grid.
4. Compute each task's MSE over its examples, supervised times, and all tape
   cells, then take the normalized task-weighted mean.
5. Backpropagate `trial_loss / number_of_base_pairs` immediately.

After all trials, gradients are clipped once and the optimizer steps once.
Different shapes are never padded or masked into one batch, and only the
largest individual trial graph needs to be resident at a time.

Free-evolution time is proportional to the sampled tape capacity, then receives
its own independent uniform perturbation:

~~~text
free_steps = round_half_up(steps_per_tape_slot * sampled_tape_slots * (1 + delta))
~~~

The defaults are 6 steps per tape slot, ±40% time variation, and a supervision
window of `round_half_up(1.5 * free_steps)`. Thus geometry, data length, and
available computation vary independently while computation remains scaled to
the actual tape.

Learning rate is defined by progress/rate anchors and either linear or cosine
interpolation. Repeating a rate across consecutive anchors creates a plateau;
changing it creates a transition. The default staged cosine schedule is:

~~~text
(0.00, 2.0e-3)  (0.35, 2.0e-3)
(0.60, 7.0e-4)  (0.75, 7.0e-4)
(0.95, 1.0e-4)  (1.00, 1.0e-4)
~~~

Cosine interpolation has zero slope at every anchor, so plateau transitions
are smooth. The same representation supports linear decay, warmup, or a
deliberate increase without a separate scheduler.

## Tasks and evaluation

Built-in string tasks are:

~~~text
copy              1011 -> 1011
bit_not           1011 -> 0100
reverse           1011 -> 1101
reverse_not       1011 -> 0010
shift_left_zero   1011 -> 0110
shift_right_zero  1011 -> 0101
gray_encode       1011 -> 1110
prefix_xor        1011 -> 1101
increment         1011 -> 1100
parity            1011 -> 1
append_0          1011 -> 10110
append_1          1011 -> 10111
~~~

The notebook defaults to a single `reverse` task for direct comparison of
convolution-only, attention-only, and combined local rules. It trains both the
shared rule and a learned read-only repeated program for 2,000 updates. The two
feature switches select the rule variant, and each variant writes to a distinct
checkpoint directory.

Evaluation uses explicit deterministic cases. Each case specifies tape
capacity, exact input length, free steps, and supervised steps. Only binary
strings of exactly that input length are evaluated; shorter strings are not
included. Cases are never capped, truncated, or resampled, and an input or
output that does not leave one blank is an error. Reports include MSE, symbol
accuracy, exact full-tape accuracy, interpreted semantic accuracy, stability
across the supervision window, and best-timestep statistics.

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

Checkpoint format 7 stores the full configuration, ordered task signatures,
model, program, optimizer groups, progress, history, and all random-generator states.
Resuming reproduces trial sampling, batches, firing masks, and perception
noise. Older formats are intentionally rejected.

This is an experimental study of programmable local computation; it does not
claim computational universality.
