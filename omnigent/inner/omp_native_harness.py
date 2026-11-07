"""``harness: omp-native`` wrap for the native OMP TUI."""

from __future__ import annotations

from fastapi import FastAPI

from omnigent.inner.executor import Executor
from omnigent.inner.omp_native_executor import OmpNativeExecutor
from omnigent.runtime.harnesses._executor_adapter import ExecutorAdapter


def _build_omp_native_executor() -> Executor:
    """Construct the native OMP bridge executor from harness spawn env."""
    return OmpNativeExecutor()


def create_app() -> FastAPI:
    """Build the ``omp-native`` harness FastAPI app."""
    adapter = ExecutorAdapter(executor_factory=_build_omp_native_executor)
    return adapter.build()
