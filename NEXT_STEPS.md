# Next steps: explicit experiment runner with Weights & Biases

> Execute these next steps only when the user tells you.

## Objective

Add a reliable Python workflow for running explicitly chosen NCPU experiments,
tracking them in Weights & Biases (W&B), periodically evaluating the configured
extrapolation cases, and saving final visualizations. Preserve the existing
Jupyter notebook unchanged.

This is not an automatic W&B hyperparameter sweep. Every experiment setup and
seed must be listed explicitly in the experiment configuration. The user or
Codex will choose those setups before they are run.

## Fixed decisions

- Use a Python runner rather than the notebook.
- Do not modify `run/run.ipynb` for this feature.
- Put editable experiment definitions in `run/experiment_config.py`, used only
  by the runner. Avoid `run/config.py` because the package already has
  `ncpu_computer.config`.
- Put execution logic in `run/run_experiments.py`.
- Run the configured `k` extrapolation `TestCase`s every 500 optimizer updates.
- These cases replace the existing deterministic base validation in this
  workflow. Do not run both validation systems.
- Do not use extrapolation metrics to select the best checkpoint.
- Select the best checkpoint by the lowest mean training MSE reported by
  `Trainer.train_step()` as `StepMetrics.loss`. This is already averaged across
  the sequential trials in one optimizer update.
- Generate GIFs only after training finishes, one GIF per trained task.
- Support both single-task and multiple-task training.
- Save each run's configuration, metrics, checkpoints, history, and final GIFs
  to W&B.
- Do not silently fall back to offline or disabled logging. If W&B is enabled
  and initialization fails, stop with a clear error.

## Experiment configuration

`run/experiment_config.py` will contain only explicit editable definitions:

- W&B entity, project, mode, and tags;
- `VALIDATION_EVERY = 500`;
- ordinary metric logging and checkpoint intervals;
- named length setups;
- explicit extrapolation cases for each length setup;
- explicit model and training setups;
- task tuples and seeds;
- final GIF specifications.

Use small frozen dataclasses or equally direct immutable structures. Do not add
a configuration framework, inheritance, automatic grids, or implicit Cartesian
products.

A named length setup keeps these values together:

```python
base_tape_slots
base_input_max_lengths
tape_variation
input_variation
free_steps_per_tape_slot
time_variation
supervision_ratio
test_cases
```

Every `test_cases` tuple may contain any positive number `k` of explicit
`TestCase` objects. The runner uses that tuple exactly.

An experiment setup identifies:

```python
name
length_setup
tasks
seeds
geometry settings
model settings
training settings
GIF inputs per task
```

## Python runner

`run/run_experiments.py` will provide:

```powershell
python run/run_experiments.py --list
python run/run_experiments.py --experiment NAME
python run/run_experiments.py --all
python run/run_experiments.py --experiment NAME --resume
```

The runner must:

1. Resolve the repository and source paths when invoked from the repository
   root or `run` directory.
2. Select only explicitly defined experiments.
3. Construct the normal `GeometryConfig`, `ModelConfig`, `TrainingConfig`, and
   `ExperimentConfig` objects.
4. Run structural validation before W&B initialization or training.
5. Create one W&B run per experiment/seed pair.
6. Train through `Trainer.train_step()` so the runner owns logging, periodic
   evaluation, checkpointing, and artifacts without changing the notebook.
7. Preserve exact checkpoint resume behavior.
8. Evaluate all configured extrapolation cases every 500 updates.
9. Save and upload final GIFs after training.
10. Finish the W&B run cleanly on success and mark failures accurately.

## Multiple-task training

The existing model already supports a shared update rule and one program tile
per task. The dataset code samples the same number of examples per task and
gives tasks equal loss weight.

For example:

```python
tasks = ("copy", "bit_not", "reverse", "parity")
```

A zero program cannot distinguish tasks that map the same input to different
outputs. For a fresh run with more than one task, require:

```python
program_mode in {"learned_read_only", "learned_mutable"}
train_program = True
```

Use `learned_read_only` as the initial multi-task baseline because it preserves
task identity throughout evolution. Keep `learned_mutable` available only when
explicitly selected. Reject ambiguous multi-task configurations before
training. Single-task runs may continue to use the fixed zero program.

## Training and checkpoint selection

For every optimizer update:

1. Call `Trainer.train_step()` once.
2. Append `StepMetrics.to_dict()` to trainer history so checkpoints retain the
   complete run history.
3. Log training metrics at the configured frequency.
4. Compare `StepMetrics.loss` with the lowest training MSE observed so far.
5. Save `best.pt` whenever this training MSE improves.
6. Save `latest.pt` at the configured checkpoint interval and at the end.

On resume, recover the best observed training MSE from saved history. Do not
reuse the current `best_validation_loss` field under a misleading meaning. The
runner owns this selection state, leaving notebook training semantics alone.

Periodic extrapolation results must not influence `best.pt` selection.

## Periodic evaluation

At every update divisible by 500, and at the final update if it is not already
such a boundary:

- run every configured extrapolation case;
- evaluate every trained task;
- use a fixed evaluation seed across runs;
- restore training behavior afterward;
- log per-task and aggregate metrics.

Use stable metric names:

```text
evaluation/<case>/<task>/mse
evaluation/<case>/<task>/semantic_accuracy
evaluation/<case>/<task>/raw_accuracy
evaluation/<case>/<task>/symbol_accuracy
evaluation/<case>/<task>/stable_accuracy
evaluation/<case>/aggregate/mse
```

Also log evaluated example count, tape capacity, input length, free steps,
supervision steps, and evaluation duration. Use complete datasets by default;
any example limit must be explicit in the experiment definition.

Although the code type remains `TestCase`, these repeatedly observed cases are
scientifically validation probes because they are visible during model and
hyperparameter selection. Do not describe them as an untouched final test.

## W&B organization and metrics

Use one W&B project so runs remain directly comparable. Store separate config
fields for:

```text
experiment_name
length_setup
tape_profile
input_profile
base_tape_slots
base_input_max_lengths
tasks
program_mode
seed
```

Use a combined length group such as:

```text
tape-5_7_8_9_11__input-3_5_6_7_8
```

Tape and input profiles remain separate W&B fields, so the UI can filter or
group by either dimension. Include task set, model size, and seed in the run
name. Do not create a separate W&B project for each length configuration.

Log training values under:

```text
train/mse
train/semantic_accuracy
train/raw_accuracy
train/learning_rate
train/gradient_norm
train/perception_noise
train/update_seconds
```

Log every sampled sequential trial's tape slots, input maximum, free steps, and
supervision steps with stable indexed keys.

## Final GIFs

After the final update:

1. Generate one GIF for every trained task.
2. Use that task's explicitly configured input string.
3. Use an explicitly selected evaluation case for tape capacity and rollout
   time.
4. Reuse the existing I/O-only visualization, fixed `[-1, +1]` scale, and
   static initial program tile.
5. Save the GIF in the run's local output directory.
6. Log it as W&B media and include it in the final artifact.

Before training, validate that every GIF task is trained, every input is valid
for that task, and every input fits the selected tape with the required blank
capacity.

## Artifacts and resume

At completion, upload one versioned artifact containing:

- `best.pt`, selected by training MSE;
- `latest.pt`;
- the resolved experiment configuration;
- metric history;
- final GIFs.

Persist the W&B run ID in the local run directory. `--resume` must restore both
the exact local checkpoint and the same W&B run. Reject mismatched experiment
configuration, task order, or seed.

Do not upload a checkpoint every 500 updates. Periodic evaluation logs metrics
only; upload the final artifact once the run completes.

## Dependency handling

Add W&B as a clearly named optional dependency for the runner and document the
installation and login commands. The core package and test suite must remain
usable without importing W&B. Import W&B only in runner-specific code. Tests
must never require authentication or network access.

## Verification

Add focused tests for:

- explicit experiment lookup and `--list`;
- rejection of unknown experiment names;
- construction and validation of every declared setup;
- arbitrary positive numbers of evaluation cases;
- evaluation scheduling at updates 500, 1000, and the final update;
- absence of base-validation calls in the new runner;
- best-checkpoint selection from training MSE only;
- periodic evaluation not changing checkpoint selection;
- single-task zero-program acceptance;
- multi-task zero-program rejection;
- valid learned-program multi-task execution;
- stable per-task and aggregate metric names;
- deterministic evaluation seeds;
- final-only GIF generation for every task;
- exact local and W&B resume identity;
- artifact contents;
- no network calls in tests;
- no modification of `run/run.ipynb`.

Run Black, Flake8, the full CPU suite, and a tiny mocked-logging smoke run. Do
not run an expensive training experiment during implementation validation.

## Implementation order

1. Add the optional W&B dependency.
2. Add `run/experiment_config.py` with a minimal valid example setup.
3. Add the explicit CLI runner.
4. Add flattened metric logging and the 500-update evaluation schedule.
5. Add training-MSE checkpoint selection and exact resume state.
6. Add final per-task GIF generation.
7. Add the final W&B artifact.
8. Add tests and concise running documentation.
9. Run formatting, linting, tests, and the mocked smoke run.

> Execute these next steps only when the user tells you.
