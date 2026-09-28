import json
from pathlib import Path


def notebook_cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert len(code_cells) == 4
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    for cell in code_cells:
        compile("".join(cell["source"]), str(path), "exec")
    return ["".join(cell["source"]) for cell in code_cells]


def test_single_clean_notebook_exposes_complete_workflow():
    run_directory = Path(__file__).parents[1] / "run"
    notebooks = list(run_directory.glob("*.ipynb"))
    assert [path.name for path in notebooks] == ["run.ipynb"]
    cells = notebook_cells(notebooks[0])
    first = cells[0]
    assert 'TASK_NAMES = ("bit_not",)' in first
    assert "STRIDE = 2" in first
    assert "VERTICAL_SPACE = 1" in first
    assert "HORIZONTAL_SPACE = 2" in first
    assert 'PROGRAM_MODE = "zero"' in first
    assert "TRAIN_RULE = True" in first
    assert "TRAIN_PROGRAM = False" in first
    assert "N_TRIALS = 3" in first
    assert "TAPE_SLOTS_MIN" in first and "TAPE_SLOTS_MAX" in first
    assert "INPUT_MAX_LENGTH_MIN" in first and "INPUT_MAX_LENGTH_MAX" in first
    assert "FREE_STEPS_MIN" in first and "FREE_STEPS_MAX" in first
    assert "SUPERVISION_RATIO = 1.6" in first
    assert "PERCEPTION_NOISE_START = 0.0" in first
    assert "TEST_CASES = (" in first
    assert "validate_experiment(config, TASK_NAMES)" in first
    assert "TapeLayout(geometry, trial.tape_slots).schema()" in first
    assert "RUN_" not in first
    assert "RUN_TRAINING =" in cells[1]
    assert "RUN_EVALUATION =" in cells[2]
    assert "evaluate_cases(" in cells[2]
    assert "RUN_VISUALIZATION =" in cells[3]
    assert "program_tile=model.programs" in cells[3]
    assert "save_gif(" in cells[3]
