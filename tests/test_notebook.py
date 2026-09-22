import json
from pathlib import Path


def notebook_cells(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    for cell in code_cells:
        compile("".join(cell["source"]), str(path), "exec")
    return ["".join(cell["source"]) for cell in code_cells]


def test_single_clean_notebook_exposes_direct_tape_workflow():
    run_directory = Path(__file__).parents[1] / "run"
    notebooks = list(run_directory.glob("*.ipynb"))
    assert [path.name for path in notebooks] == ["run.ipynb"]
    cells = notebook_cells(notebooks[0])
    source = "\n".join(cells)
    first = cells[0]
    assert 'TASK_NAME = "bit_not"' in first
    assert '"reverse": reverse_task' in first
    assert "TRAIN_MAX_LENGTH = 7" in first
    assert "TRAIN_TAPE_SLOTS = TRAIN_MAX_LENGTH" in first
    assert "TEST_LENGTH = TRAIN_MAX_LENGTH" in first
    assert "TEST_TAPE_SLOTS = TRAIN_TAPE_SLOTS" in first
    assert "STRIDE = 2" in first
    assert 'INPUT_MODE = "mutable"' in first
    assert "PROGRAM_CHANNELS = 1" in first
    assert "COMPUTATION_CHANNELS = 3" in first
    assert "GeometryConfig(" in first
    assert "ModelConfig(" in first
    assert "TrainingConfig(" in first
    assert "validate_experiment(config, train_dataset)" in first
    assert "train_layout.schema()" in first
    assert "RUN_" not in first
    assert "RUN_TRAINING =" in cells[1]
    assert "RUN_VALIDATION =" in cells[2]
    assert "RUN_VISUALIZATION =" in cells[3]
    assert "save_gif(" in source
    assert "include_shorter=False" in source
    assert "render_tape(encoded)" in source
    assert "validation_dataset = TaskDataset.from_task(" in cells[2]
    assert "TEST_TAPE_SLOTS" in cells[2]
    assert "EXTRAPOLATION_LENGTH" not in source
    assert "RUN_EVALUATION" not in source
    assert "ood_geometry" not in source
