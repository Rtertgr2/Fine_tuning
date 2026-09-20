"""Split behaviour (T1.8): group-aware, deterministic, stratified."""

from __future__ import annotations

import pytest

from backend.app.services.datasets import split_examples
from backend.validators.base import Example

from tests.conftest import msg, read_call


def _ex(ex_id: str, category: str, group: str | None, i: int = 0):
    return Example(
        id=ex_id,
        category=category,
        messages=[
            msg("user", f"งานที่ {i}"),
            msg("assistant", read_call(f"f{i}.py")),
            msg("tool", "content"),
            msg("assistant", f"เสร็จงานที่ {i}"),
        ],
        group_id=group,
    )


def test_group_never_straddles():
    examples = []
    for g in range(5):
        for k in range(2):
            examples.append(_ex(f"ex_{g}_{k}", "tool", f"group{g}", g * 2 + k))
    train, val = split_examples(examples, val_ratio=0.3, seed=42)

    train_groups = {e.group_id for e in train}
    val_groups = {e.group_id for e in val}
    assert not (train_groups & val_groups), "a group leaked across train/val"
    assert len(train) + len(val) == len(examples)


def test_same_seed_same_split():
    examples = [_ex(f"ex_{i}", "tool", f"g{i}", i) for i in range(10)]
    a = split_examples(examples, val_ratio=0.3, seed=7)
    b = split_examples(examples, val_ratio=0.3, seed=7)
    assert [e.id for e in a[0]] == [e.id for e in b[0]]
    assert [e.id for e in a[1]] == [e.id for e in b[1]]


def test_stratified_by_category():
    examples = []
    for i in range(6):
        examples.append(_ex(f"tool_{i}", "tool", f"gt{i}", i))
    for i in range(6):
        examples.append(_ex(f"plan_{i}", "plan", f"gp{i}", i))
    train, val = split_examples(examples, val_ratio=0.5, seed=1)
    val_cats = {e.category for e in val}
    assert val_cats == {"tool", "plan"}, "each category should be represented in val"


def test_single_group_goes_to_train():
    examples = [_ex("ex_1", "tool", "only-group", 0)]
    train, val = split_examples(examples, val_ratio=0.3, seed=1)
    assert len(train) == 1 and len(val) == 0


def test_solo_examples_are_own_groups():
    examples = [_ex(f"ex_{i}", "tool", None, i) for i in range(4)]
    train, val = split_examples(examples, val_ratio=0.5, seed=3)
    assert len(train) + len(val) == 4
    assert len(val) >= 1
