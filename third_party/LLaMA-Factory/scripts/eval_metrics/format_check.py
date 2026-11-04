"""Parse model response text into a 6-DoF pose list."""

import json
import re
from typing import Optional

Pose = list[float]


def parse_pose(response: str) -> Optional[Pose]:
    """Try to extract a list of 6 floats from *response*.

    Handles both bare ``[1,2,3,4,5,6]`` and markdown-fenced code blocks.
    Returns ``None`` when parsing fails.
    """
    text = response.strip()

    # Strip markdown code fences if present
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    text = text.strip()

    # Try JSON parse first (handles most well-formed outputs)
    try:
        obj = json.loads(text)
        if isinstance(obj, list) and len(obj) == 6:
            return [float(v) for v in obj]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Fallback: extract the first bracket-delimited list of numbers
    m = re.search(r"\[([^\[\]]+)\]", text)
    if m:
        try:
            values = [float(v.strip()) for v in m.group(1).split(",")]
            if len(values) == 6:
                return values
        except ValueError:
            pass

    return None
