"""Guards the default harness wiring for the Polly and Debby example bundles.

Polly and Debby defaulted to Claude-family harnesses until 2026-11-06, when
Claude was cancelled and OMP became the default brain/worker harness. This
module asserts that both bundles now resolve to the headless ``omp`` harness by
default, that the matching ``omp`` sub-agent configs exist, and that explicit
harness overrides remain authoritative.
"""

from __future__ import annotations

from pathlib import Path

from omnigent.spec import load

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _sub_agents(bundle: str) -> dict[str, object]:
    spec = load(_REPO_ROOT / "examples" / bundle)
    return {sa.name: sa for sa in (getattr(spec, "sub_agents", None) or [])}


def _config(sub_agent: object) -> dict[str, object]:
    executor = getattr(sub_agent, "executor", None)
    config = getattr(executor, "config", None)
    if isinstance(config, dict):
        return config
    return {}


def _executor_config(bundle: str) -> dict[str, object]:
    spec = load(_REPO_ROOT / "examples" / bundle)
    executor = getattr(spec, "executor", None)
    config = getattr(executor, "config", None)
    if isinstance(config, dict):
        return config
    return {}


def test_polly_brain_defaults_to_omp_harness() -> None:
    """Polly's orchestrator brain runs on the headless ``omp`` harness."""
    cfg = _executor_config("polly")
    assert cfg.get("harness") == "omp"
    # No smart_routing_harness remains from the old claude-sdk configuration.
    assert "smart_routing_harness" not in cfg


def test_polly_default_sub_agent_is_omp() -> None:
    """Polly declares ``omp`` as the default sub-agent and keeps optional workers."""
    spec = load(_REPO_ROOT / "examples" / "polly")
    assert spec.tools.agents[0] == "omp"
    subs = _sub_agents("polly")
    assert "omp" in subs
    assert _config(subs["omp"]).get("harness") == "omp"
    # Optional legacy workers remain available for explicit overrides.
    for name in ("claude_code", "codex", "opencode", "cursor", "hermes", "agy", "pi"):
        assert name in subs, name


def test_debby_brain_defaults_to_omp_harness() -> None:
    """Debby's orchestrator brain runs on the headless ``omp`` harness."""
    cfg = _executor_config("debby")
    assert cfg.get("harness") == "omp"
    assert "smart_routing_harness" not in cfg


def test_debby_heads_are_omp_and_gpt() -> None:
    """Debby has exactly two heads: ``omp`` and ``gpt``."""
    spec = load(_REPO_ROOT / "examples" / "debby")
    assert sorted(spec.tools.agents) == ["gpt", "omp"]
    subs = _sub_agents("debby")
    assert _config(subs["omp"]).get("harness") == "omp"
    assert _config(subs["gpt"]).get("harness") == "codex"


def test_polly_omp_agent_config_exists_and_uses_omp_harness() -> None:
    """The bundled Polly OMP sub-agent config exists and declares ``harness: omp``."""
    cfg_path = _REPO_ROOT / "examples" / "polly" / "agents" / "omp" / "config.yaml"
    assert cfg_path.exists()
    spec = load(cfg_path.parent)
    assert spec.executor.config.get("harness") == "omp"


def test_debby_omp_agent_config_exists_and_uses_omp_harness() -> None:
    """The bundled Debby OMP sub-agent config exists and declares ``harness: omp``."""
    cfg_path = _REPO_ROOT / "examples" / "debby" / "agents" / "omp" / "config.yaml"
    assert cfg_path.exists()
    spec = load(cfg_path.parent)
    assert spec.executor.config.get("harness") == "omp"


def test_debby_stays_opencode_free() -> None:
    """Debby does not re-introduce an opencode head (reverted in #1295)."""
    subs = _sub_agents("debby")
    assert "opencode" not in subs
    config_text = (_REPO_ROOT / "examples" / "debby" / "config.yaml").read_text(encoding="utf-8")
    assert "opencode" not in config_text.lower()
    # No head re-introduces opencode-native via a harness override either.
    for sub in subs.values():
        assert "opencode-native" not in (_config(sub).get("allowed_harnesses") or [])
