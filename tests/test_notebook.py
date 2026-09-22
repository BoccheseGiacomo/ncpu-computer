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


def test_single_clean_notebook_exposes_programmed_tape_workflow():
    run_directory = Path(__file__).parents[1] / "run"
    notebooks = list(run_directory.glob("*.ipynb"))
    assert [path.name for path in notebooks] == ["run.ipynb"]
    cells = notebook_cells(notebooks[0])
    source = "\n".join(cells)
    first = cells[0]
    assert (
        'TASK_NAMES = ("copy", "bit_not", "reverse", "parity", "append_0", "append_1")'
        in first
    )
    assert "TRAIN_MAX_LENGTH = 6" in first
    assert "TEST_LENGTH = 8" in first
    assert "TRAIN_TAPE_SLOTS = 10" in first
    assert "TEST_TAPE_SLOTS = 10" in first
    assert "STRIDE = 2" in first
    assert 'IO_MODE = "separate"' in first
    assert 'INPUT_MODE = "mutable"' in first
    assert 'PROGRAM_PLACEMENT = "grid"' in first
    assert "PROGRAM_MUTABLE = False" in first
    assert "PROGRAM_INIT_STD = 0.02" in first
    assert "PROGRAM_WEIGHT_DECAY = 1e-4" in first
    assert "PROGRAM_CHANNELS = 1" in first
    assert "BATCH_SIZE_PER_TASK" in first
    assert "GeometryConfig(" in first
    assert "ModelConfig(" in first
    assert "TrainingConfig(" in first
    assert "validate_experiment(config, train_datasets)" in first
    assert "layout.schema()" in first
    assert "TRAIN_TAPE_SLOTS != TEST_TAPE_SLOTS" in first
    assert "RUN_" not in first
    assert "RUN_TRAINING =" in cells[1]
    assert "RUN_VALIDATION =" in cells[2]
    assert "RUN_VISUALIZATION =" in cells[3]
    assert 'VISUALIZATION_TASK = "reverse"' in cells[3]
    assert "save_gif(" in source
    assert "include_shorter=False" in source
    assert "model.initial_state(layout.render_tape(encoded), task_indices)" in source
    assert "validation_datasets = MultiTaskDataset.from_tasks(" in cells[2]
    assert "evaluate_tasks(" in cells[2]
    assert "TEST_TAPE_SLOTS" in cells[2]
    assert "task_name=VISUALIZATION_TASK" in cells[3]
