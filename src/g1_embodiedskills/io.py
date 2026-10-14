"""Small atomic experiment records; binary artifacts remain local."""
import hashlib
import json
from pathlib import Path


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    def convert(x):
        if hasattr(x, "detach"):
            return x.detach().cpu().tolist()
        if hasattr(x, "tolist"):
            return x.tolist()
        if isinstance(x, Path):
            return str(x)
        raise TypeError(type(x).__name__)
    content = json.dumps(value, ensure_ascii=False, indent=2, default=convert, allow_nan=False)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(content + "\n", encoding="utf-8")
    temporary.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


class RunFinished(Exception):
    def __init__(self, code=0):
        self.code = code
