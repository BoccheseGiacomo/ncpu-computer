# Plan: one-dimensional convolution-attention NCA

This document defines a deliberately simple next architecture. It is a plan,
not an implementation. Do not execute it until the user explicitly asks.

## Goal

Replace the two-dimensional strided grid with a one-dimensional tape in which
every physical position is one logical tape cell. Each cell contains a dynamic
channel vector. A shared recurrent rule updates those vectors using:

1. a narrow one-dimensional convolution;
2. wider local, non-causal attention;
3. an additive residual update.

The purpose is to remove unnecessary geometry while retaining local recurrent
computation and permitting faster communication than convolution alone.

## Direct tape interface

The tape tensor has shape:

```text
batch x channels x tape_slots
```

There is no stride, spacer, border lane, or unused physical cell. Tape position
`i` is tensor position `i`.

The channel vector is:

```text
program channels | one shared I/O channel | computation channels
```

The default total width is 32 channels:

```text
4 program + 1 shared I/O + 27 computation = 32
```

The direct ternary interface remains unchanged:

```text
binary 0 -> -1
blank B  ->  0
binary 1 -> +1
```

Only the shared I/O channel is initialized from the input. Computation channels
start at zero. Input and output use the same mutable I/O channel.

At training time, ordinary MSE supervises the I/O channel at every tape
position. Target blanks are zero and are supervised through the end of the
tape. Computation channels are latent and receive no direct target.

At inference, decode the I/O channel with the existing strict thresholds:

```text
value >  +0.333 -> 1
value <  -0.333 -> 0
otherwise       -> B
```

There is no learned symbol embedding, categorical loss, or tied symbol
readout. The complete channel vector is the dynamic embedding used internally.

## Program representation

For a single task, the default program is fixed zero and read-only.

For multiple tasks, each task owns one learned program vector with shape:

```text
program_channels
```

That vector is broadcast identically to every tape position. Use a learned
read-only program as the initial multi-task baseline. Do not introduce a
periodic spatial program or absolute coordinates.

## Recurrent update

For state `h` at one timestep:

```text
conv_delta = convolution_branch(h)
attn_delta = local_attention_branch(h)
delta      = conv_delta + attn_delta
h_next     = h + mutable_mask * delta
```

Both branches return exactly the full channel dimension. Their outputs update
the dynamic embeddings directly. Do not concatenate them into another large
fusion network in the first version.

Read-only program channels are protected by the existing mutable mask. The I/O
and computation channels always remain mutable.

The rule is shared across every tape position and every recurrent timestep.

## Convolution branch

Use one learned local block:

```text
Conv1d(channels, hidden_size, kernel_size=2*r+1, padding=r)
ReLU
Conv1d(hidden_size, channels, kernel_size=1, bias=False)
```

Defaults:

```text
convolution radius r = 1
hidden size          = 96
kernel size          = 3
```

The spatial convolution is allowed to distinguish left from right. Do not tie
opposite kernel offsets. This branch supplies directional local structure.

Zero-initialize the final pointwise projection so the initial convolution
delta is exactly zero.

## Attention branch

Use local multi-head attention over the full dynamic cell embeddings.

Defaults:

```text
attention radius k = 4
window size        = 2*k+1 = 9
heads              = 4
head dimension     = 8
embedding width    = 32
```

Require `r < k`, an even head dimension, and total channels divisible by the
number of heads.

Queries, keys, and values are bias-free projections of the complete state:

```text
Q = Wq h
K = Wk h
V = Wv h
```

The attention window for position `i` is bidirectional and non-causal:

```text
i-k, ..., i-1, i, i+1, ..., i+k
```

Include the center position in the first implementation. Mask only positions
outside the finite tape. Never mask `B`, because blank is a real tape symbol.

Implement local windows directly. Do not construct a full `length x length`
attention matrix. Complexity should be `O(batch * length * k * channels)`.

## Symmetric distance RoPE

Use one positional mechanism only: symmetric distance RoPE.

For a query at `i` and key at `j`, define:

```text
distance = abs(j - i)
```

The score for one head is:

```text
score(i, j) = Q_i dot Rotate(K_j, distance) / sqrt(head_dimension)
```

The rotary sine and cosine angles use the non-negative distance. Consequently,
left and right positions at the same distance receive exactly the same
rotation:

```text
i-2 and i+2 -> identical distance rotation
```

Use ordinary fixed RoPE frequencies initially. Rotate keys only for the
pairwise score; do not rotate values. Do not add learned absolute positions,
signed relative biases, NoPE modes, or alternative positional systems in the
first implementation.

The attention branch therefore knows content and distance but not direction.
The convolution branch remains responsible for left/right distinction.

Concatenate the attended heads and apply a bias-free output projection back to
the full channel dimension. Zero-initialize that output projection so the
initial attention delta is exactly zero.

## Normalization and magnitude

Do not normalize the recurrent state in the first version.

Specifically, do not add:

- LayerNorm;
- RMSNorm;
- query/key unit normalization;
- soft RMS capping;
- magnitude-dependent update gates.

Magnitude is part of the dynamic embedding and must remain available to the
rule. Standard scaled dot-product attention, zero-initialized residual outputs,
gradient clipping, and the existing configurable state bound are sufficient
for the baseline.

Keep the current hard state bound configurable, with the existing default. It
is a safety limit, not a normalization step. Record state RMS and maximum
absolute state during diagnostics. Add a more specialized stabilization method
only if measurements show actual growth or saturation.

## Attention sinks

Do not add a sink-specific mechanism initially.

Classic attention sinks are less likely here because attention is local and
non-causal: no token is visible to every later position. A local high-magnitude
attractor remains possible, but it should be measured before modifying the
architecture.

During diagnostics, record:

- mean attention entropy;
- maximum attention probability;
- attention mass by absolute distance;
- state RMS and maximum absolute value.

Do not add null tokens, forced attention dropout, center exclusion, or custom
attention gates in the baseline.

## Remaining NCA behavior

Preserve the useful behavior of the current implementation:

- recurrent residual evolution;
- configurable computation time;
- long supervision windows;
- full-tape blank supervision;
- variable tape and input lengths;
- sequential differently shaped trials with gradient accumulation;
- configurable fire rate, defaulting to `1.0` for this baseline;
- configurable read-only or mutable program channels;
- direct ternary decoding;
- explicit deterministic evaluation cases;
- exact checkpoint resume.

Use zero padding for convolution. Attention masks positions outside the tape.
Do not add circular wrapping in the first version.

## Initial scientific comparisons

First establish one complete convolution-attention baseline. After it works,
compare it against a convolution-only ablation by disabling the attention
branch. Do not implement several attention position modes at once.

The first useful comparison is therefore:

```text
A. 1D convolution only
B. 1D convolution + non-causal local attention with distance RoPE
```

Keep state width, data, training times, optimizer, and evaluation cases equal.

Evaluate at least:

- bit-NOT, as a local sanity check;
- reversal, as the main communication test;
- the existing in-range, longer-tape, and longer-input cases;
- multi-task training only after single-task correctness is established.

## Implementation route

When implementation is authorized:

1. Preserve the current branch or checkpoint before architectural replacement.
2. Replace the 2D tape layout with direct rank-three 1D state tensors.
3. Implement direct input rendering and I/O extraction without stride or
   spacers.
4. Implement the convolution branch and verify zero-initialized identity
   evolution.
5. Implement efficient local window extraction.
6. Implement symmetric distance RoPE and non-causal local attention.
7. Add the two full-width branch deltas and apply the mutable mask.
8. Adapt training, evaluation, inference, checkpointing, and visualization to
   1D states.
9. Remove the obsolete 2D geometry and program-tile code rather than keeping
   compatibility paths.
10. Add the convolution-only ablation flag only after the complete baseline is
    correct.

## Required tests

Test:

- tape width equals logical tape capacity;
- direct ternary input and full-tape I/O extraction;
- blank cells initialize to exact zero with a zero program;
- the attention window reads both left and right neighbors;
- one attention step cannot access positions beyond radius `k`;
- `-d` and `+d` use identical rotary angles;
- different absolute tape coordinates with the same distance use the same
  rotation;
- attention uses complete channel vectors rather than only the I/O channel;
- output deltas have the full channel dimension;
- zero-initialized branches make the initial rule exactly the identity;
- read-only program channels never change;
- I/O and computation channels can change;
- variable tape lengths work without padding batches together;
- no full quadratic attention matrix is created;
- checkpoint resume remains exact;
- inference and GIF visualization show only the decoded I/O tape by default.

Run formatting, linting, the full CPU test suite, and only tiny smoke runs. Do
not run an expensive training experiment as implementation validation.

## Explicit non-goals for the first version

Do not initially add:

- learned symbol embeddings;
- categorical output loss;
- separate input and output channels;
- absolute positions;
- signed RoPE;
- learned positional bias in addition to RoPE;
- global attention;
- causal attention;
- LayerNorm or RMSNorm;
- soft query/key normalization;
- attention-sink tokens or special gates;
- multiple stacked attention layers per NCA timestep;
- a large fusion MLP after the two branches.

These remain possible ablations only after the simplest complete architecture
has been implemented and measured.
