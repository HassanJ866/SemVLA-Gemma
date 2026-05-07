"""
Prompt templates for the three brain tasks.

Each function returns a list of messages in the format expected by
transformers / Gemma4 chat templates (role + content as list of typed dicts).
Images are represented as {"type": "image"} entries; text as {"type": "text", "text": ...}.
"""

import json


SYSTEM_MSG = (
    "You are a robot perception module. Always respond with valid JSON only. "
    "Do not include any explanation or extra text outside the JSON object."
)


def grounding_prompt(instruction: str) -> list[dict]:
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": (
                    f"Task: {instruction}\n\n"
                    "Locate the target object mentioned in the task. "
                    'Output a JSON object with keys "object" (string) and '
                    '"bbox" ([x1, y1, x2, y2] pixel coordinates).'
                )},
            ],
        },
    ]


def parsing_prompt(bboxes: list[dict]) -> list[dict]:
    bbox_str = json.dumps(bboxes, separators=(",", ":"))
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": (
                    f"Detected objects: {bbox_str}\n\n"
                    "Output the bidirectional spatial scene graph for all objects. "
                    'Use JSON with key "triplets": a list of [subject, relation, object] arrays. '
                    "Relations must be from: is_left_of, is_right_of, is_above, is_below, "
                    "is_in_front_of, is_behind, is_on, is_under, is_inside, contains. "
                    "Include both directions for every spatial relation."
                )},
            ],
        },
    ]


def task_synthesis_prompt(src_name: str, src_bbox: list, dst_name: str,
                           dst_bbox: list, src_graph: list) -> list[dict]:
    src_graph_str = json.dumps(src_graph, separators=(",", ":"))
    return [
        {"role": "system", "content": [{"type": "text", "text": SYSTEM_MSG}]},
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": (
                    f"Source object: {src_name}  bbox: {src_bbox}  (the object to pick up)\n"
                    f"Destination object: {dst_name}  bbox: {dst_bbox}  (where to place it)\n"
                    f"Source spatial context: {src_graph_str}\n\n"
                    "Describe a pick-and-place task that moves the source object to the destination. "
                    'Output JSON with key "task": a natural language instruction string. '
                    'Example: {"task": "pick up the black bowl on the ramekin and place it on the plate"}'
                )},
            ],
        },
    ]


def format_training_sample(record: dict) -> dict:
    """
    Convert a JSONL record to (messages, target_text) for supervised fine-tuning.
    Returns {"messages": [...], "target": str}
    """
    task = record["task_type"]
    target = record["output"]  # already a JSON string

    if task == "grounding":
        messages = grounding_prompt(record["instruction"])
    elif task == "parsing":
        bboxes = record.get("bboxes", [])
        messages = parsing_prompt(bboxes)
    elif task == "task_synthesis":
        messages = task_synthesis_prompt(
            record["src_name"],
            record["src_bbox"],
            record["dst_name"],
            record["dst_bbox"],
            record.get("src_graph", []),
        )
    else:
        raise ValueError(f"Unknown task_type: {task}")

    return {"messages": messages, "target": target}
