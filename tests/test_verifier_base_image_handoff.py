"""A verifier whose Dockerfile declares `ARG BASE_IMAGE` is built on the rebuilt task image (MaxxBench)."""

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from pier.trial.trial import Trial


def _fake_trial(tests_dir: Path, image: str | None):
    seen = {}

    async def verify_in(env_config, build_context, **kwargs):
        seen["context"] = build_context
        compose = build_context / "docker-compose.yaml"
        seen["compose"] = json.loads(compose.read_text()) if compose.exists() else None
        seen["dockerfile"] = (build_context / "Dockerfile").read_text()
        return "verified"

    fake = SimpleNamespace(
        _verifier_env_build_context=lambda step_cfg: tests_dir,
        _environment=SimpleNamespace(_env_vars=SimpleNamespace(main_image_name=image)),
        _verify_in_separate_environment=verify_in,
    )
    return fake, seen


def _call(fake):
    return asyncio.run(Trial._verify_with_separate_environment(
        fake, None, key="k", step_cfg=None, verifier_env=None, artifacts_dir=Path("/nonexistent")))


def test_base_image_arg_hands_over_the_rebuilt_image_in_a_copy(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "Dockerfile").write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n")
    fake, seen = _fake_trial(tests_dir, "pier-task-main:abc")
    assert _call(fake) == "verified"
    assert seen["context"] != tests_dir
    assert seen["compose"] == {"services": {"main": {"build": {"args": {"BASE_IMAGE": "pier-task-main:abc"}}}}}
    assert not (tests_dir / "docker-compose.yaml").exists()          # the task tree is never edited
    assert not seen["context"].exists()                              # the copy is gone after verify


def test_without_the_arg_the_tests_dir_is_used_as_is(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "Dockerfile").write_text("FROM public.ecr.aws/x/y:v1\n")
    fake, seen = _fake_trial(tests_dir, None)
    assert _call(fake) == "verified"
    assert seen["context"] == tests_dir and seen["compose"] is None


def test_the_arg_without_a_rebuilt_image_is_refused(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "Dockerfile").write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n")
    fake, _ = _fake_trial(tests_dir, None)
    with pytest.raises(RuntimeError, match="rebuilt Docker task image"):
        _call(fake)


def test_an_existing_verifier_compose_file_is_never_replaced(tmp_path):
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "Dockerfile").write_text("ARG BASE_IMAGE\nFROM ${BASE_IMAGE}\n")
    (tests_dir / "docker-compose.yaml").write_text("services: {}\n")
    fake, _ = _fake_trial(tests_dir, "img")
    with pytest.raises(RuntimeError, match="existing verifier Compose"):
        _call(fake)
