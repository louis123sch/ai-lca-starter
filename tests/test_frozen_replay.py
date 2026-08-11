from types import SimpleNamespace

import pytest

import ai_lca.frozen_replay as frozen_replay


class FakeStore:
    def __init__(self):
        self.calls = []

    def record_processes(self, doi, structure):
        self.calls.append((doi, structure))


def processor_with_store():
    processor = object.__new__(frozen_replay.FrozenControlProcessor)
    processor.store = FakeStore()
    return processor


def test_frozen_structure_mixin_loads_existing_structure(monkeypatch, tmp_path):
    path = tmp_path / "extraction" / "structure.json"
    path.parent.mkdir(parents=True)
    path.write_text("{}", encoding="utf-8")
    sentinel = SimpleNamespace(processes=[])
    seen = {}

    def fake_load_model(received_path, model):
        seen["path"] = received_path
        seen["model"] = model
        return sentinel

    monkeypatch.setattr(frozen_replay, "_load_model", fake_load_model)
    processor = processor_with_store()
    result = frozen_replay.FrozenStructureMixin._structure(
        processor,
        "10.0000/example",
        "hash",
        None,
        tmp_path,
    )

    assert result is sentinel
    assert seen["path"] == path
    assert seen["model"] is frozen_replay.ForegroundStructure
    assert processor.store.calls == [("10.0000/example", sentinel)]


def test_frozen_structure_mixin_refuses_missing_structure(tmp_path):
    processor = processor_with_store()
    with pytest.raises(FileNotFoundError):
        frozen_replay.FrozenStructureMixin._structure(
            processor,
            "10.0000/example",
            "hash",
            None,
            tmp_path,
        )


def test_both_ab_processors_resolve_structure_to_freeze_mixin():
    assert frozen_replay.FrozenControlProcessor._structure is frozen_replay.FrozenStructureMixin._structure
    assert frozen_replay.FrozenRoutedProcessor._structure is frozen_replay.FrozenStructureMixin._structure
