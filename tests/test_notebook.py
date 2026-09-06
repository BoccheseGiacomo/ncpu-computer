import json
from pathlib import Path


def notebook_source(path):
    notebook = json.loads(path.read_text(encoding="utf-8"))
    code_cells = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
    assert code_cells
    assert all(cell["execution_count"] is None for cell in code_cells)
    assert all(cell["outputs"] == [] for cell in code_cells)
    return "\n".join("".join(cell["source"]) for cell in code_cells)


def test_notebooks_are_clean_and_have_explicit_training_switches():
    run_directory = Path(__file__).parents[1] / "run"
    sources = {
        path.name: notebook_source(path) for path in run_directory.glob("*.ipynb")
    }
    assert set(sources) == {"run.ipynb", "simple_binary_tasks.ipynb"}
    assert all("RUN_TRAINING = False" in source for source in sources.values())
    assert "RUN_8BIT_EVALUATION = False" in sources["run.ipynb"]
    assert all("RUN_VISUALIZATION = False" in source for source in sources.values())
    assert all("GeometryConfig(" in source for source in sources.values())
    assert all("ModelConfig(" in source for source in sources.values())
    assert all("TrainingConfig(" in source for source in sources.values())
    assert all("print(layout.schema())" in source for source in sources.values())
    assert all("save_gif(" in source for source in sources.values())
    simple = sources["simple_binary_tasks.ipynb"]
    assert 'TASK_NAME = "reverse"' in simple
    assert '"bit_not": bitwise_not_task' in simple
