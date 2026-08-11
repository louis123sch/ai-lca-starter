from pathlib import Path

from . import inventory_replay


if __name__ == "__main__":
    inventory_replay.CANARY_MANIFEST = Path(
        "benchmarks/corpus_development_v1/micro.json"
    )
    inventory_replay.main()
