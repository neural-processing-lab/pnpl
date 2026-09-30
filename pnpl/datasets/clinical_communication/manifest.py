"""Immutable benchmark metadata; independent of recording downloads."""
import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path

DATA = Path(__file__).with_name("data")

def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def asset_path(name):
    """Locate and verify a bundled binary or JSON asset."""
    path = DATA / name
    if name != "provenance.json":
        expected = load("provenance.json")["files"][name]
        if sha256(path) != expected:
            raise ValueError(f"Clinical benchmark asset checksum mismatch: {name}")
    return path

@lru_cache(None)
def load(name):
    """Read a bundled asset after checking its frozen checksum."""
    payload = asset_path(name).read_bytes()
    if name.endswith(".gz"):
        payload = gzip.decompress(payload)
    return json.loads(payload)

def normalize_partition(partition):
    aliases = {"validation": "val", "valid": "val", "development": "dev"}
    partition = aliases.get(partition, partition)
    if partition not in {"train", "val", "test", "dev"}:
        raise ValueError("partition must be train, val, test, or dev")
    return partition
