import json
from pathlib import Path


def test_notebook_is_clean_and_has_explicit_run_switches():
    notebook_path = Path(__file__).parents[1] / "run" / "run.ipynb"
    notebook = json.loads(notebook_path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    source = "\n".join("".join(cell["source"]) for cell in code_cells)
    assert "RUN_TRAINING = False" in source
    assert "RUN_8BIT_EVALUATION = False" in source
