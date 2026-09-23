# ncpu-computer

`ncpu-computer` studies computation as the repeated application of one small,
shared neural cellular automaton (NCA) rule. The grid provides working memory,
and repeated local updates provide computation time.

The model uses a general string interface over `{0, 1, B}`. `B` is blank and
fills unused capacity; it may also separate fields in structured inputs. An
external deterministic codec can serialize integers, Boolean sequences, tuples,
or other finite data into this alphabet. The NCA itself always receives the
same direct scalar representation:

| Symbol | State value | Meaning |
|---|---:|---|
| `0` | `-1` | bit zero |
| `B` | `0` | blank or separator |
| `1` | `+1` | bit one |

There is no learned input encoder or output decoder.

## Tape geometry

The grid contains two aligned horizontal tapes with `N` logical slots each:
input above, output below. Both use the same origin and centre-to-centre stride
`s` (default 2). Left, right, top, bottom, and inter-tape spacing are
independently configurable:

```text
height = top + 1 + inter_tape_rows + 1 + bottom
width  = left + (N - 1)s + 1 + right

working space above

left | I0 . I1 . I2 . ... . I(N-1) | right

working space between tapes

left | O0 . O1 . O2 . ... . O(N-1) | right

working space below
```

Only the marked cells are logical tape positions. Other cells are NCA working
space initialized to zero. Optional `wrap_y` joins the top and bottom edges
for perception; horizontal edges never wrap. Without `wrap_y`, the configured
zero, reflect, or replicate padding applies on both axes.

Strings are left-aligned. There is no compulsory leading blank, terminator
slot, or additional tail position. A shorter string is represented by implicit
blank zeros through the fixed tape capacity:

```text
empty: BBBB
0:     0BBB
1:     1BBB
00:    00BB
01:    01BB
```

A full-length string occupies all `N` positions and needs no final blank.

## Channels

Input and output use separate rows of the same mutable channel. The default
five-channel state is:

| Channel | Role | Mutable |
|---:|---|---|
| 0 | zero program channel | no |
| 1 | shared input/output state | yes |
| 2–4 | computation state | yes |

At time zero, the encoded input is injected into the upper tape. The lower
tape, program, and computation state start at zero. The entire I/O channel is
mutable during evolution; only the program channel is read-only and zero.

## Local computation

For grid state `X_t`, let `M` be one on mutable channels and zero on read-only
channels. One deterministic update is:

```text
D_t     = U_theta(P(X_t))
Y_t     = clip(X_t + M * D_t)
X_(t+1) = M * Y_t + (1 - M) * X_t
```

`P` applies a configurable bank of depthwise `3 x 3` perception kernels to
every state channel. `U_theta` is a shared two-layer `1 x 1` network with a ReLU
hidden layer. Its output projection starts at zero, so the initial dynamics are
exactly the identity map. Optional gates, stochastic cell firing, state
clipping, padding modes, and learned perception kernels remain configurable.

The same parameters are used at every cell and timestep. Parameter count does
not depend on tape length, although boundary distance can still affect the
dynamics and must be tested empirically.

## Tasks and variable lengths

The initial notebook supports bitwise NOT and reversal. Each training set
contains the empty string and every binary string through the selected maximum
length, ordered by length and then lexicographically.

String length and tape capacity are independent. `TRAIN_MAX_LENGTH` controls
which strings occur in training, while `TRAIN_TAPE_SLOTS` controls the physical
training grid. Every shorter input and target is blank-padded to that full
capacity, and every target blank contributes to the MSE.

For a four-cell tape, bitwise NOT begins:

```text
BBBB -> BBBB
0BBB -> 1BBB
1BBB -> 0BBB
00BB -> 11BB
01BB -> 10BB
10BB -> 01BB
11BB -> 00BB
```

Reversal uses the same inputs, with targets such as `01BB -> 10BB`. This makes
length and blank handling part of the training distribution rather than
training only on strings that fill the tape.

Post-training validation uses one exact `TEST_LENGTH` encoded into
`TEST_TAPE_SLOTS`. Keeping the training and test capacities equal isolates
length extrapolation within a fixed geometry: choose a capacity larger than
`TRAIN_MAX_LENGTH`, then validate at a longer length within that same tape.
Changing `TEST_TAPE_SLOTS` instead also tests generalization to a new grid
width and boundary distance.

## Training and readout

The target is used only to calculate loss. It is never injected during
evolution. The upper input tape is not supervised after initialization. After
`F` free computation steps, the lower output tape is supervised at
every state in a window of `S` steps:

```text
F + 1, F + 2, ..., F + S
```

The sole objective is ordinary mean squared error over examples, supervised
timesteps, and every logical position of the lower tape:

```text
loss = mean((output_tape - target_tape)^2)
```

Every padded blank is therefore an explicit target value of zero. There are no
additional terminator or tail objectives.

Inference gathers all output positions in parallel. External decoding uses the
literal threshold `0.333`:

```text
value >  +0.333 -> 1
value <  -0.333 -> 0
otherwise       -> B
```

The full raw tape is available before an optional interpreter stops at a blank
or separates multiple fields. Evaluation reports MSE, symbol accuracy,
whole-tape exact accuracy, interpreted accuracy, and stability across the
supervision window.

## Running experiments

On Windows:

```powershell
conda env create -f environment.yml
conda activate slackenv
pip install -e ".[dev,notebook]"
pytest -q
jupyter lab run/run.ipynb
```

[`run/run.ipynb`](run/run.ipynb) also works without an editable installation
when launched from either the repository root or `run/`. Its first code cell
exposes the shared task, tape capacities, lengths, geometry, channel counts,
vertical wrapping, local rule, optimizer, and supervision settings. It always
runs the structural checks and prints the physical layout before any optional
action.

Training, checkpoint loading, post-training validation, and visualization have
controls in their respective cells. Post-training validation evaluates one
configurable exact input length. The visualization shows only the shared I/O
channel on a fixed `-1` to `+1` color scale and marks both logical tapes.

## Repository structure

```text
src/ncpu_computer/
  config.py       geometry, channel roles, model, and training settings
  tape.py         direct ternary codec, strided layout, and interpreters
  tasks.py        variable-length string tasks and tensor datasets
  model.py        perception and residual NCA update rule
  training.py     full-tape MSE, optimization, and checkpoints
  evaluation.py   tensorized metrics and inference
  validation.py   representation and gradient invariant checks
  visualize.py    annotated output-channel trajectory GIFs
run/run.ipynb     configurable bit-NOT and reversal workflow
tests/            CPU regression tests
```

Checkpoint format 4 belongs to this direct scalar, two-lane shared-channel model.
Incompatible representation formats are rejected explicitly.

## Direction

The current program channel is read-only and zero. A later stage can initialize
it from a task index while keeping the direct input/output tape unchanged. The
shared rule can then be trained across tasks, tape sizes, and computation times
to test reusable local computation without conflating that question with a
learned data representation.

This project does not claim computational universality. It is an experimental
system for testing when small local learned rules support stable, scalable, and
eventually programmable computation.
