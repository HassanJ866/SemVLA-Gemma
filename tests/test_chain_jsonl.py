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
    task_synthesis_prompt,
    format_training_sample,
)


def _content_types(msg):
    return [c["type"] for c in msg["content"]]

def _content_text(msg):
    return " ".join(c["text"] for c in msg["content"] if c["type"] == "text")


def test_grounding_prompt_structure():
    msgs = grounding_prompt("pick up the red bowl")
    assert len(msgs) == 2
    assert msgs[0]["role"] == "system"
    assert msgs[1]["role"] == "user"
    assert "image" in _content_types(msgs[1])
    assert "pick up the red bowl" in _content_text(msgs[1])


def test_parsing_prompt_structure():
    bboxes = [{"name": "bowl", "bbox": [10, 20, 50, 60]}]
    msgs = parsing_prompt(bboxes)
    assert msgs[1]["role"] == "user"
    assert "image" in _content_types(msgs[1])
    text = _content_text(msgs[1])
    assert "bowl" in text
    assert "triplets" in text


def test_task_synthesis_prompt_structure():
    src_graph = [["akita_black_bowl_1", "is_on_top_of", "glazed_rim_porcelain_ramekin_1"]]
    msgs = task_synthesis_prompt(
        "akita_black_bowl_1", [94, 47, 128, 78],
        "plate_1", [79, 13, 116, 48],
        src_graph,
    )
    assert msgs[1]["role"] == "user"
    assert "image" in _content_types(msgs[1])
    text = _content_text(msgs[1])
    assert "akita_black_bowl_1" in text
    assert "plate_1" in text
    assert "task" in text


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
        "output": '{"triplets": [["bowl", "is_on_top_of", "plate"]]}',
    }
    out = format_training_sample(rec)
    target = json.loads(out["target"])
    assert "triplets" in target
    assert len(target["triplets"]) == 1


def test_format_task_synthesis_sample():
    rec = {
        "task_type": "task_synthesis",
        "image": "test/frame.png",
        "src_name": "akita_black_bowl_2",
        "src_bbox": [94, 47, 128, 78],
        "dst_name": "plate_1",
        "dst_bbox": [79, 13, 116, 48],
        "src_graph": [["akita_black_bowl_2", "is_on_top_of", "glazed_rim_porcelain_ramekin_1"]],
        "output": '{"task": "pick up the black bowl and place it on the plate"}',
    }
    out = format_training_sample(rec)
    assert "messages" in out and "target" in out
    target = json.loads(out["target"])
    assert "task" in target
    assert isinstance(target["task"], str)
    assert len(target["task"]) > 0


def test_task_synthesis_output_json_validity():
    valid_outputs = [
        '{"task": "pick up the black bowl and place it on the plate"}',
        '{"task": "pick up the black bowl and place it on the ramekin"}',
        '{"task": "pick up the cookie box and place it on the stove"}',
        '{"task": "pick up the ramekin and place it on the plate"}',
    ]
    for s in valid_outputs:
        obj = json.loads(s)
        assert isinstance(obj, dict)
        assert "task" in obj
        assert isinstance(obj["task"], str)


def test_json_output_validity():
    for s in [
        '{"object": "bowl", "bbox": [10, 20, 50, 60]}',
        '{"triplets": [["bowl", "is_on_top_of", "plate"], ["plate", "is_below_of", "bowl"]]}',
        '{"task": "pick up the black bowl on the ramekin and place it on the plate"}',
    ]:
        obj = json.loads(s)
        assert isinstance(obj, dict)


def test_unknown_task_type_raises():
    rec = {
        "task_type": "invalid_task",
        "output": "{}",
    }
    try:
        format_training_sample(rec)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass


if __name__ == "__main__":
    test_grounding_prompt_structure()
    test_parsing_prompt_structure()
    test_task_synthesis_prompt_structure()
    test_format_grounding_sample()
    test_format_parsing_sample()
    test_format_task_synthesis_sample()
    test_task_synthesis_output_json_validity()
    test_json_output_validity()
    test_unknown_task_type_raises()
    print("All chain/JSONL tests passed.")
