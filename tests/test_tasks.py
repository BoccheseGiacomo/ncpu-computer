import pytest
import torch

from ncpu_computer.tasks import (
    StringExample,
    StringTask,
    TaskDataset,
    addition_task,
    bitwise_not_task,
    parity_task,
    reverse_task,
    semantic_correct,
)


def test_binary_tasks_include_empty_and_all_shorter_strings_in_order():
    reverse = reverse_task(2)
    assert reverse.examples == (
        StringExample("", ""),
        StringExample("0", "0"),
        StringExample("1", "1"),
        StringExample("00", "00"),
        StringExample("01", "10"),
        StringExample("10", "01"),
        StringExample("11", "11"),
    )
    bit_not = [
        (example.input, example.target) for example in bitwise_not_task(2).examples
    ]
    assert bit_not == [
        ("", ""),
        ("0", "1"),
        ("1", "0"),
        ("00", "11"),
        ("01", "10"),
        ("10", "01"),
        ("11", "00"),
    ]
    parity = {example.input: example.target for example in parity_task(1).examples}
    assert parity == {"": "0", "0": "0", "1": "1"}
    assert len(addition_task(2).examples) == 16


def test_exact_length_tasks_exclude_empty_and_shorter_strings():
    inputs = [
        example.input for example in reverse_task(2, include_shorter=False).examples
    ]
    assert inputs == ["00", "01", "10", "11"]


def test_task_validation_rejects_ambiguous_targets():
    with pytest.raises(ValueError, match="single-output"):
        StringTask("bad", (StringExample("1", "1B1"),))
    with pytest.raises(ValueError, match="single-B"):
        StringTask("bad", (StringExample("1", "1BB1"),), "multiple")


def test_dataset_uses_direct_values_and_blank_fills_full_capacity():
    task = StringTask("copy", (StringExample("101", "10"), StringExample("", "")))
    dataset = TaskDataset.from_task(task, tape_slots=4)
    assert dataset.inputs.tolist() == [
        [1.0, -1.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    assert dataset.targets.tolist() == [
        [1.0, -1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ]
    assert dataset.target_lengths.tolist() == [2, 0]


def test_full_length_output_needs_no_extra_terminator_slot():
    dataset = TaskDataset.from_task(
        StringTask("copy", (StringExample("111", "111"),)), tape_slots=3
    )
    prediction = dataset.targets.to(torch.int8)
    assert semantic_correct(
        prediction,
        dataset.targets.to(torch.int8),
        dataset.target_lengths,
        "single",
    ).item()


def test_semantic_correct_requires_available_blank_but_ignores_later_tail():
    target = torch.tensor([[1, -1, 0, 0]], dtype=torch.int8)
    lengths = torch.tensor([2])
    predictions = torch.tensor([[[1, -1, 0, 1], [1, -1, 1, 0]]], dtype=torch.int8)
    assert semantic_correct(predictions, target, lengths, "single").tolist() == [
        [True, False]
    ]
