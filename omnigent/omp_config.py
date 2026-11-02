"""Profile-safe access to OMP's persistent model configuration.

This module is deliberately a small boundary around the OMP CLI.  It never
reads or writes OMP configuration files directly, and the runner seam keeps
callers' tests independent from an installed OMP binary or user profile.
"""

from __future__ import annotations

import json
import re
import subprocess
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, TypeAlias

from omnigent.inner.omp_executor import NativeModelOption

OmpRunner: TypeAlias = Callable[[list[str]], subprocess.CompletedProcess[str]]

_IDENTIFIER_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")
_THINKING_LEVELS = frozenset({"off", "minimal", "low", "medium", "high", "xhigh"})


class OmpConfigError(RuntimeError):
    """Base error for an OMP configuration command failure."""

    def __init__(self, command_class: str, message: str) -> None:
        super().__init__(f"OMP {command_class}: {message}")
        self.command_class = command_class


class OmpCommandError(OmpConfigError):
    """Raised when OMP cannot run a configuration command successfully."""

    def __init__(self, command_class: str, returncode: int | None, message: str) -> None:
        super().__init__(command_class, message)
        self.returncode = returncode


class OmpConfigParseError(OmpConfigError):
    """Raised when OMP returns malformed structured configuration data."""


class OmpValidationError(ValueError):
    """Raised before an invalid model or thinking-level value reaches OMP."""


def _default_runner(argv: list[str]) -> subprocess.CompletedProcess[str]:
    """Run one OMP argv command without a shell or exception-on-nonzero mode."""
    return subprocess.run(argv, capture_output=True, check=False, text=True)


def _validate_identifier(value: str, *, name: str) -> str:
    """Return a strict OMP identifier or reject values that cannot be model parts."""
    if not isinstance(value, str) or not _IDENTIFIER_RE.fullmatch(value):
        raise OmpValidationError(
            f"OMP {name} must be a non-empty identifier containing only letters, numbers, "
            "'.', '_', or '-'."
        )
    return value


def _parse_json_object(raw: str, *, command_class: str, label: str) -> dict[str, Any]:
    """Parse one non-empty OMP JSON object without echoing command output in errors."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OmpConfigParseError(command_class, f"returned malformed {label} JSON.") from exc
    if not isinstance(value, dict):
        raise OmpConfigParseError(
            command_class, f"returned a {label} value that is not an object."
        )
    return value


def _parse_model_catalog(raw: str) -> list[NativeModelOption]:
    """Normalize OMP's structured model output into native model-picker options."""
    command_class = "models"
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise OmpConfigParseError(command_class, "returned malformed catalog JSON.") from exc

    entries: object
    if isinstance(payload, list):
        entries = payload
    elif isinstance(payload, dict):
        entries = payload.get("models")
    else:
        raise OmpConfigParseError(
            command_class, "returned a catalog that is not an object or list."
        )
    if not isinstance(entries, list):
        raise OmpConfigParseError(command_class, "returned a catalog without a models list.")

    options: list[NativeModelOption] = []
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise OmpConfigParseError(
                command_class, "returned a catalog with a non-object model entry."
            )
        provider = entry.get("provider")
        model_id = entry.get("id")
        if not isinstance(provider, str) or not isinstance(model_id, str):
            raise OmpConfigParseError(
                command_class, "returned a catalog with an incomplete model entry."
            )
        try:
            _validate_identifier(provider, name="provider")
            _validate_identifier(model_id, name="model id")
        except OmpValidationError as exc:
            raise OmpConfigParseError(
                command_class, "returned a catalog with an invalid model identifier."
            ) from exc
        display_name = entry.get("displayName") or entry.get("name") or model_id
        provider_name = entry.get("providerName")
        aliases = entry.get("aliases", [])
        if not isinstance(display_name, str) or (
            provider_name is not None and not isinstance(provider_name, str)
        ) or not isinstance(aliases, list) or not all(isinstance(alias, str) for alias in aliases):
            raise OmpConfigParseError(
                command_class, "returned a catalog with invalid model metadata."
            )
        options.append(
            NativeModelOption(
                provider=provider,
                model_id=model_id,
                display_name=display_name,
                provider_name=provider_name,
                aliases=list(aliases),
            )
        )
    return options


@dataclass(frozen=True)
class OmpProfileConfig:
    """OMP profile selection and an injectable argv-only command boundary."""

    profile: str | None = None
    runner: OmpRunner = _default_runner
    executable: str = "omp"

    def __post_init__(self) -> None:
        if not isinstance(self.profile, str | type(None)):
            raise OmpValidationError("OMP profile must be a string or None.")
        if not isinstance(self.executable, str) or not self.executable:
            raise OmpValidationError("OMP executable must be a non-empty string.")

    def list_models(self) -> list[NativeModelOption]:
        """Return OMP's structured model catalog for the selected profile."""
        result = self._run("models", "models", "--json")
        return _parse_model_catalog(result.stdout)

    def persist_model(self, provider: str, model_id: str) -> None:
        """Persist a default model without replacing unrelated model-role settings."""
        provider = _validate_identifier(provider, name="provider")
        model_id = _validate_identifier(model_id, name="model id")
        current_roles = self._get_model_roles()
        updated_roles = dict(current_roles)
        updated_roles["default"] = _model_role_value(
            current_roles.get("default"), provider=provider, model_id=model_id
        )
        payload = json.dumps(updated_roles, separators=(",", ":"), sort_keys=True)
        self._run("config set", "config", "set", "modelRoles", payload)

    def persist_thinking_level(self, level: str) -> None:
        """Persist one documented OMP default thinking level."""
        if not isinstance(level, str) or level not in _THINKING_LEVELS:
            allowed = ", ".join(sorted(_THINKING_LEVELS))
            raise OmpValidationError(f"OMP thinking level must be one of: {allowed}.")
        self._run("config set", "config", "set", "defaultThinkingLevel", level)

    def _get_model_roles(self) -> dict[str, Any]:
        """Read the complete role map before replacing only its default entry."""
        result = self._run("config get", "config", "get", "modelRoles", "--json")
        return _parse_json_object(result.stdout, command_class="config get", label="modelRoles")

    def _run(self, command_class: str, *args: str) -> subprocess.CompletedProcess[str]:
        """Run one OMP argv command and convert start/nonzero failures to typed errors."""
        argv = [self.executable]
        if self.profile:
            argv.extend(("--profile", self.profile))
        argv.extend(args)
        try:
            result = self.runner(argv)
        except (OSError, subprocess.SubprocessError) as exc:
            raise OmpCommandError(command_class, None, "could not start the command.") from exc
        if result.returncode != 0:
            raise OmpCommandError(
                command_class,
                result.returncode,
                f"failed with exit status {result.returncode}.",
            )
        return result


def _model_role_value(existing: object, *, provider: str, model_id: str) -> object:
    """Preserve OMP's existing role-value representation when updating its model."""
    if isinstance(existing, Mapping):
        updated = dict(existing)
        updated["provider"] = provider
        updated["modelId" if "modelId" in existing else "model"] = model_id
        return updated
    return f"{provider}/{model_id}"


def list_omp_models(
    profile: str | None, *, runner: OmpRunner = _default_runner, executable: str = "omp"
) -> list[NativeModelOption]:
    """List OMP models through an injectable profile-scoped CLI boundary."""
    return OmpProfileConfig(profile=profile, runner=runner, executable=executable).list_models()


def persist_omp_model(
    profile: str | None,
    provider: str,
    model_id: str,
    *,
    runner: OmpRunner = _default_runner,
    executable: str = "omp",
) -> None:
    """Persist the selected default model through ``omp config set`` only."""
    OmpProfileConfig(profile=profile, runner=runner, executable=executable).persist_model(
        provider, model_id
    )


def persist_omp_thinking_level(
    profile: str | None,
    level: str,
    *,
    runner: OmpRunner = _default_runner,
    executable: str = "omp",
) -> None:
    """Persist the selected default thinking level through ``omp config set`` only."""
    OmpProfileConfig(profile=profile, runner=runner, executable=executable).persist_thinking_level(
        level
    )


__all__ = [
    "OmpCommandError",
    "OmpConfigError",
    "OmpConfigParseError",
    "OmpProfileConfig",
    "OmpRunner",
    "OmpValidationError",
    "list_omp_models",
    "persist_omp_model",
    "persist_omp_thinking_level",
]
