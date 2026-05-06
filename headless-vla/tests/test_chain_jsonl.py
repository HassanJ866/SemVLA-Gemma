"""
Tests for the 3-task JSONL format and prompt templates.
These tests do NOT require a GPU or a trained model.
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.brain.prompts import (
    grounding_prompt,
    parsing_prompt,
    action_prompt,
    format_training_sample,
)
from models.middleware.enums import semantic_action_to_ids, ids_to_semantic_action


def test_grounding_prompt_structure():
    msgs = grounding_prompt("pick up the red bowl")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "pick up the red bowl" in msgs[1]["content"]
    assert "<image>" in msgs[1]["content"]


def test_parsing_prompt_structure():
    bboxes = [{"name": "bowl", "bbox": [10, 20, 50, 60]}]
    msgs = parsing_prompt(bboxes)
    assert msgs[1]["role"] == "user"
    assert "bowl" in msgs[1]["content"]
    assert "triplets" in msgs[1]["content"]


def test_action_prompt_structure():
    sg = {"triplets": [["bowl", "is_on", "plate"]]}
    msgs = action_prompt("pick up the bowl", sg,
                         [0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert msgs[1]["role"] == "user"
    assert "axis" in msgs[1]["content"]
    assert "gripper" in msgs[1]["content"]


def test_format_grounding_sample():
    rec = {
        "task_type": "grounding",
        "image": "test/frame.png",
        "instruction": "pick up the bowl",
        "output": '{"object": "bowl", "bbox": [10, 20, 50, 60]}',
    }
    out = format_training_sample(rec)
    assert "messages" in out and "target" in out
    target = json.loads(out["target"])
    assert target["object"] == "bowl"
    assert len(target["bbox"]) == 4


def test_format_parsing_sample():
    rec = {
        "task_type": "parsing",
        "image": "test/frame.png",
        "bboxes": [{"name": "bowl", "bbox": [10, 20, 50, 60]}],
        "output": '{"triplets": [["bowl", "is_on", "plate"]]}',
    }
    out = format_training_sample(rec)
    target = json.loads(out["target"])
    assert "triplets" in target
    assert len(target["triplets"]) == 1


def test_format_action_sample():
    rec = {
        "task_type": "action",
        "instruction": "pick up the bowl",
        "scene_graph": {"triplets": []},
        "proprio": [0.0] * 9,
        "output": '{"axis": "Z", "direction": "negative", "magnitude": "small", "gripper": "keep"}',
    }
    out = format_training_sample(rec)
    target = json.loads(out["target"])
    assert target["axis"] == "Z"
    assert target["gripper"] == "keep"


def test_enum_roundtrip():
    actions = [
        {"axis": "X", "direction": "positive",  "magnitude": "small",  "gripper": "open"},
        {"axis": "Y", "direction": "negative",  "magnitude": "medium", "gripper": "close"},
        {"axis": "Z", "direction": "positive",  "magnitude": "large",  "gripper": "keep"},
    ]
    for action in actions:
        ids = semantic_action_to_ids(action)
        assert len(ids) == 4
        recovered = ids_to_semantic_action(ids)
        assert recovered == action, f"Roundtrip failed: {action} -> {ids} -> {recovered}"


def test_json_output_validity():
    for s in [
        '{"object": "bowl", "bbox": [10, 20, 50, 60]}',
        '{"triplets": [["bowl", "is_on", "plate"], ["plate", "is_under", "bowl"]]}',
        '{"axis": "X", "direction": "positive", "magnitude": "small", "gripper": "keep"}',
    ]:
        obj = json.loads(s)
        assert isinstance(obj, dict)


if __name__ == "__main__":
    test_grounding_prompt_structure()
    test_parsing_prompt_structure()
    test_action_prompt_structure()
    test_format_grounding_sample()
    test_format_parsing_sample()
    test_format_action_sample()
    test_enum_roundtrip()
    test_json_output_validity()
    print("All chain/JSONL tests passed.")
