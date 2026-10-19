import json
import time

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from nanochat import checkpoint_manager


def _save_distributed_checkpoint(rank, checkpoint_dir, init_method, release_rank_one):
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
    )
    try:
        if rank == 1:
            release_rank_one.wait()
        checkpoint_manager.save_checkpoint(
            checkpoint_dir,
            step=75,
            model_data={"weight": torch.tensor([1.0])},
            optimizer_data={"rank": torch.tensor([rank])},
            meta_data={"step": 75},
            rank=rank,
        )
    finally:
        dist.destroy_process_group()


def test_save_checkpoint_publishes_metadata_last(tmp_path, monkeypatch):
    events = []
    real_replace = checkpoint_manager.os.replace

    def record_replace(source, destination):
        events.append(checkpoint_manager.os.path.basename(destination))
        real_replace(source, destination)

    monkeypatch.setattr(checkpoint_manager.os, "replace", record_replace)
    monkeypatch.setattr(
        checkpoint_manager,
        "_checkpoint_barrier",
        lambda: events.append("barrier"),
    )

    checkpoint_manager.save_checkpoint(
        tmp_path,
        step=25,
        model_data={"weight": torch.tensor([1.0])},
        optimizer_data={"momentum": torch.tensor([2.0])},
        meta_data={"step": 25},
        rank=0,
    )

    assert events == [
        "model_000025.pt",
        "optim_000025_rank0.pt",
        "barrier",
        "meta_000025.json",
        "barrier",
    ]
    assert json.loads((tmp_path / "meta_000025.json").read_text()) == {"step": 25}
    assert not list(tmp_path.glob("*.tmp-*"))


def test_failed_artifact_is_never_committed(tmp_path, monkeypatch):
    real_save = checkpoint_manager.torch.save

    def fail_optimizer(value, destination):
        if isinstance(destination, checkpoint_manager.os.PathLike):
            name = checkpoint_manager.os.fspath(destination)
        else:
            name = getattr(destination, "name", "")
        if "optim_" in name:
            raise OSError("simulated interrupted optimizer write")
        return real_save(value, destination)

    monkeypatch.setattr(checkpoint_manager.torch, "save", fail_optimizer)

    with pytest.raises(OSError, match="interrupted optimizer write"):
        checkpoint_manager.save_checkpoint(
            tmp_path,
            step=50,
            model_data={"weight": torch.tensor([1.0])},
            optimizer_data={"momentum": torch.tensor([2.0])},
            meta_data={"step": 50},
            rank=0,
        )

    assert not (tmp_path / "meta_000050.json").exists()
    assert not list(tmp_path.glob("*.tmp-*"))


def test_model_only_checkpoint_does_not_wait_for_distributed_ranks(tmp_path, monkeypatch):
    def unexpected_barrier():
        raise AssertionError("model-only checkpoints must not enter a distributed barrier")

    monkeypatch.setattr(checkpoint_manager, "_checkpoint_barrier", unexpected_barrier)

    checkpoint_manager.save_checkpoint(
        tmp_path,
        step=60,
        model_data={"weight": torch.tensor([1.0])},
        optimizer_data=None,
        meta_data={"step": 60},
        rank=0,
    )

    assert (tmp_path / "model_000060.pt").exists()
    assert json.loads((tmp_path / "meta_000060.json").read_text()) == {"step": 60}


def test_metadata_waits_for_every_distributed_rank(tmp_path):
    context = mp.get_context("spawn")
    release_rank_one = context.Event()
    init_method = f"file://{tmp_path / 'distributed-init'}"
    processes = [
        context.Process(
            target=_save_distributed_checkpoint,
            args=(rank, str(tmp_path), init_method, release_rank_one),
        )
        for rank in range(2)
    ]

    for process in processes:
        process.start()

    deadline = time.monotonic() + 15
    rank_zero_files_exist = False
    while time.monotonic() < deadline:
        rank_zero_files_exist = (
            (tmp_path / "model_000075.pt").exists()
            and (tmp_path / "optim_000075_rank0.pt").exists()
        )
        if rank_zero_files_exist:
            break
        time.sleep(0.05)

    assert rank_zero_files_exist
    assert not (tmp_path / "meta_000075.json").exists()

    release_rank_one.set()
    for process in processes:
        process.join(timeout=15)
        assert not process.is_alive()
        assert process.exitcode == 0

    assert (tmp_path / "optim_000075_rank1.pt").exists()
    assert json.loads((tmp_path / "meta_000075.json").read_text()) == {"step": 75}
