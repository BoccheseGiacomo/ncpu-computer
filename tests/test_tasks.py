import pytest
import torch

from ncpu_computer.tasks import (
    StringExample,
    StringTask,
    TaskDataset,
    addition_task,
    parity_task,
    reverse_task,
    semantic_correct,
)


def test_builtin_tasks_have_exact_string_semantics():
    addition = addition_task(2)
    examples = {(example.input, example.target) for example in addition.examples}
    assert len(examples) == 16
    assert ("11B10", "101") in examples
    assert ("0B0", "0") in examples
    assert reverse_task(2).examples[-1] == StringExample("11", "11")
    parity = {example.input: example.target for example in parity_task(2).examples}
    assert parity == {"0": "0", "1": "1", "00": "0", "01": "1", "10": "1", "11": "0"}


def test_task_validation_rejects_ambiguous_targets():
    with pytest.raises(ValueError, match="single-output"):
        StringTask("bad", (StringExample("1", "1B1"),))
    with pytest.raises(ValueError, match="single-B"):
        StringTask("bad", (StringExample("1", "1BB1"),), "multiple")


def test_dataset_pads_targets_and_constructs_single_masks():
    task = StringTask("copy", (StringExample("101", "10"),))
    dataset = TaskDataset.from_task(task, tape_slots=6)
    assert dataset.inputs.tolist() == [[1.0, -1.0, 1.0, 0.0, 0.0, 0.0]]
    assert dataset.targets.tolist() == [[1.0, -1.0, 0.0, 0.0, 0.0, 0.0]]
    assert dataset.target_lengths.tolist() == [2]
    assert dataset.terminator_mask.tolist() == [
        [False, False, True, False, False, False]
    ]
    assert dataset.tail_mask.tolist() == [[False, False, False, True, True, True]]


def test_multiple_output_terminator_is_two_blanks():
    task = StringTask(
        "split", (StringExample("1011", "10B11"),), output_mode="multiple"
    )
    dataset = TaskDataset.from_task(task, tape_slots=8)
    assert dataset.terminator_mask.tolist() == [
        [False, False, False, False, False, True, True, False]
    ]
    assert dataset.tail_mask.tolist() == [
        [False, False, False, False, False, False, False, True]
    ]


def test_semantic_correct_requires_output_and_terminator_but_ignores_tail():
    target = torch.tensor([[1, -1, 0, 0]], dtype=torch.int8)
    lengths = torch.tensor([2])
    predictions = torch.tensor([[[1, -1, 0, 1], [1, -1, 1, 0]]], dtype=torch.int8)
    assert semantic_correct(predictions, target, lengths, "single").tolist() == [
        [True, False]
    ]


def test_dataset_rejects_insufficient_terminator_capacity():
    task = StringTask("copy", (StringExample("1", "111"),))
    with pytest.raises(ValueError, match="terminator"):
        TaskDataset.from_task(task, tape_slots=3)
