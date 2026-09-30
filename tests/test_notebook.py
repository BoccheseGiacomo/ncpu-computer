import json
from pathlib import Path


def notebook_cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 5
    for cell in code_cells:
        compile("".join(cell["source"]), str(path), "exec")
    return ["".join(cell["source"]) for cell in code_cells]


def test_single_notebook_exposes_complete_workflow():
    run_directory = Path(__file__).parents[1] / "run"
    notebooks = list(run_directory.glob("*.ipynb"))
    assert [path.name for path in notebooks] == ["run.ipynb"]
    cells = notebook_cells(notebooks[0])
    first = cells[0]
    assert 'TASKS = (("reverse", 1.0),)' in first
    assert "TASK_NAMES, TASK_WEIGHTS = validate_task_specs(TASKS)" in first
    assert 'EXPERIMENT_NAME = "reverse_local_rule_comparison"' in first
    assert "STRIDE = 2" in first
    assert "VERTICAL_SPACE = 1" in first
    assert "HORIZONTAL_SPACE = 2" in first
    assert "PROGRAM_START = 1" in first
    assert 'PROGRAM_MODE = "learned_read_only"' in first
    assert "PROGRAM_CHANNELS = 2" in first
    assert "COMPUTATION_CHANNELS = 4" in first
    assert "HIDDEN_SIZE = 96" in first
    assert "CONVOLUTION_ENABLED = True" in first
    assert "ATTENTION_ENABLED = True" in first
    assert "ATTENTION_RADIUS = 2" in first
    assert "ATTENTION_DIM = 16" in first
    assert "ATTENTION_HEADS = 2" in first
    assert "ATTENTION_DISTANCE_BIAS = True" in first
    assert "ATTENTION_QK_CAP = 2.0" in first
    assert "TRAIN_RULE = True" in first
    assert "TRAIN_PROGRAM = True" in first
    assert "UPDATES = 2000" in first
    assert "BATCH_SIZE_PER_TASK = 64" in first
    assert "BASE_TAPE_SLOTS = (5, 7, 8, 9, 11)" in first
    assert "BASE_INPUT_MAX_LENGTHS = (3, 5, 6, 7, 8)" in first
    assert "TAPE_VARIATION = 0.20" in first
    assert "INPUT_VARIATION = 0.20" in first
    assert "FREE_STEPS_PER_TAPE_SLOT = 6.0" in first
    assert "TIME_VARIATION = 0.20" in first
    assert "SUPERVISION_RATIO = 1.5" in first
    assert "LR_POINTS = (" in first
    assert "(0.00, 2e-3)" in first
    assert "(0.65, 6e-4)" in first
    assert "(1.0, 1e-5)" in first
    assert 'LR_INTERPOLATION = "cosine"' in first
    assert "LEARNING_RATE" not in first
    assert "WARMUP_UPDATES" not in first
    assert "TEST_CASES = (" in first
    assert 'TestCase("train_large", 11, 8, 66, 99)' in first
    assert 'TestCase("longer_tape", 14, 8, 84, 126)' in first
    assert 'TestCase("longer_tape_input", 14, 11, 84, 126)' in first
    assert "validate_experiment(config, TASK_NAMES)" in first
    assert '"conv_attention" if CONVOLUTION_ENABLED and ATTENTION_ENABLED' in first
    assert 'else "convolution_only" if CONVOLUTION_ENABLED' in first
    assert 'else "attention_only"' in first
    assert "layout = TapeLayout(geometry, trial.tape_slots)" in first
    assert 'f"{layout.schema()}"' in first
    assert "RUN_" not in first
    assert "RUN_TRAINING =" in cells[1]
    assert "LOSS_RUNNING_MEAN = 100" in cells[1]
    assert "metrics.validation_accuracies" in cells[1]
    assert 'loss_axis.set_yscale("log")' in cells[1]
    assert 'learning_rate_axis.set_yscale("log")' in cells[1]
    assert "loss_axis.set_ylim(1e-6, 1.0)" in cells[1]
    assert "RUN_EVALUATION =" in cells[2]
    assert "evaluate_cases(" in cells[2]
    assert "RUN_VISUALIZATION =" in cells[3]
    assert "VISUALIZATION_TASK = TASK_NAMES[0]" in cells[3]
    assert "program_tile=model.programs" in cells[3]
    assert "save_gif(" in cells[3]
