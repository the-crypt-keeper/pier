import asyncio
import shutil
from collections.abc import Coroutine
from typing import Any

from pier.models.job.config import RetryConfig
from pier.models.trial.config import TrialConfig
from pier.models.trial.result import TrialResult
from pier.trial.hooks import HookCallback, TrialEvent
from pier.utils.logger import logger


class TrialQueue:
    """
    Handles orchestration of concurrent trials.

    Receives TrialConfigs, creates Trial objects internally, runs them
    with retry logic, and returns TrialResult tasks. Concurrency is
    bounded by an asyncio.Semaphore (no backends) or by N serial workers
    (with backends — one worker per backend, trials pinned 1:1).
    """

    def __init__(
        self,
        n_concurrent: int,
        retry_config: RetryConfig | None = None,
        hooks: dict[TrialEvent, list[HookCallback]] | None = None,
        backends: list[str] | None = None,
    ):
        if hooks is None:
            hooks = {event: [] for event in TrialEvent}
        else:
            for event in TrialEvent:
                hooks.setdefault(event, [])

        self._n_concurrent = n_concurrent
        self._retry_config = retry_config if retry_config is not None else RetryConfig()
        self._hooks = hooks
        self._logger = logger.getChild(__name__)
        self._semaphore = asyncio.Semaphore(n_concurrent)
        self._backends = backends or []

    def add_hook(self, event: TrialEvent, callback: HookCallback) -> "TrialQueue":
        """Register a callback for a trial lifecycle event and return the queue."""
        self._hooks[event].append(callback)
        return self

    def on_trial_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a queued trial starts."""
        return self.add_hook(TrialEvent.START, callback)

    def on_environment_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a trial environment starts."""
        return self.add_hook(TrialEvent.ENVIRONMENT_START, callback)

    def on_agent_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a trial agent starts."""
        return self.add_hook(TrialEvent.AGENT_START, callback)

    def on_verification_started(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when trial verification starts."""
        return self.add_hook(TrialEvent.VERIFICATION_START, callback)

    def on_trial_ended(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a queued trial ends."""
        return self.add_hook(TrialEvent.END, callback)

    def on_trial_cancelled(self, callback: HookCallback) -> "TrialQueue":
        """Register a callback that runs when a queued trial is cancelled."""
        return self.add_hook(TrialEvent.CANCEL, callback)

    def _should_retry_exception(self, exception_type: str) -> bool:
        """Check if an exception should trigger a retry."""
        if (
            self._retry_config.exclude_exceptions
            and exception_type in self._retry_config.exclude_exceptions
        ):
            self._logger.debug(
                f"Exception {exception_type} is in exclude_exceptions, not retrying"
            )
            return False

        if (
            self._retry_config.include_exceptions
            and exception_type not in self._retry_config.include_exceptions
        ):
            self._logger.debug(
                f"Exception {exception_type} is not in include_exceptions, not retrying"
            )
            return False

        return True

    def _calculate_backoff_delay(self, attempt: int) -> float:
        """Calculate the backoff delay for a retry attempt."""
        delay = self._retry_config.min_wait_sec * (
            self._retry_config.wait_multiplier**attempt
        )
        return min(delay, self._retry_config.max_wait_sec)

    def _setup_hooks(self, trial) -> None:
        """Wire queue-level hooks to the trial."""
        for event, hooks in self._hooks.items():
            for hook in hooks:
                trial.add_hook(event, hook)

    @staticmethod
    def _pin_backend(trial_config: TrialConfig, backend_url: str) -> TrialConfig:
        """Return a copy of the config with OPENAI_BASE_URL pinned."""
        config = trial_config.model_copy(deep=True)
        config.agent.env["OPENAI_BASE_URL"] = backend_url
        config.agent.env["OPENAI_API_BASE"] = backend_url
        return config

    async def _execute_trial_with_retries(
        self, trial_config: TrialConfig
    ) -> TrialResult:
        """Execute a trial with retry logic."""
        from pier.trial.trial import Trial

        for attempt in range(self._retry_config.max_retries + 1):
            trial = await Trial.create(trial_config)
            self._setup_hooks(trial)
            result = await trial.run()

            if result.exception_info is None:
                return result

            if not self._should_retry_exception(result.exception_info.exception_type):
                self._logger.debug(
                    "Not retrying trial because the exception is not in "
                    "include_exceptions or the maximum number of retries has been "
                    "reached"
                )
                return result
            if attempt == self._retry_config.max_retries:
                self._logger.debug(
                    "Not retrying trial because the maximum number of retries has been "
                    "reached"
                )
                return result

            shutil.rmtree(trial.trial_dir, ignore_errors=True)

            delay = self._calculate_backoff_delay(attempt)

            self._logger.debug(
                f"Trial {trial_config.trial_name} failed with exception "
                f"{result.exception_info.exception_type}. Retrying in "
                f"{delay:.2f} seconds..."
            )

            await asyncio.sleep(delay)

        raise RuntimeError(
            f"Trial {trial_config.trial_name} produced no result. This should never "
            "happen."
        )

    async def _run_trial(self, trial_config: TrialConfig) -> TrialResult:
        """Execute a single trial, acquiring the semaphore for concurrency control."""
        async with self._semaphore:
            return await self._execute_trial_with_retries(trial_config)

    def submit(self, trial_config: TrialConfig) -> Coroutine[Any, Any, TrialResult]:
        """
        Return a coroutine that executes one trial.

        The caller decides how to schedule it (await, gather, TaskGroup).
        """
        return self._run_trial(trial_config)

    def submit_batch(
        self, configs: list[TrialConfig]
    ) -> list[Coroutine[Any, Any, TrialResult]]:
        """
        Return coroutines for multiple trials, ordered to match `configs`.

        Without backends: each coroutine acquires a semaphore slot.
        With backends: trials are assigned to workers round-robin. Each
        worker runs its trials serially on one backend. Each coroutine
        still returns one TrialResult — the caller sees no difference.
        """
        if not self._backends:
            return [self.submit(config) for config in configs]

        return self._submit_batch_pinned(configs)

    def _submit_batch_pinned(
        self, configs: list[TrialConfig]
    ) -> list[Coroutine[Any, Any, TrialResult]]:
        """Split trials across backends. One worker per backend, serial within.

        Each trial gets a Future. Workers resolve futures as they finish.
        Each returned coroutine just awaits its own future — the caller's
        TaskGroup drives them all, and results come back in order.
        """
        n_workers = len(self._backends)

        # Assign each trial to a worker and create a future for its result
        futures: list[asyncio.Future[TrialResult]] = []
        worker_items: list[list[tuple[TrialConfig, asyncio.Future[TrialResult]]]] = [
            [] for _ in range(n_workers)
        ]

        loop = asyncio.get_event_loop()
        for i, config in enumerate(configs):
            fut: asyncio.Future[TrialResult] = loop.create_future()
            futures.append(fut)
            if n_workers == 1:
                wid = 0
            else:
                wid = i % n_workers
            worker_items[wid].append((config, fut))

        # Start all workers as background tasks
        for wid in range(n_workers):
            asyncio.ensure_future(
                self._worker(wid, self._backends[wid], worker_items[wid])
            )

        # Return one coroutine per trial that awaits its future
        async def _await_future(fut: asyncio.Future[TrialResult]) -> TrialResult:
            return await fut

        return [_await_future(f) for f in futures]

    async def _worker(
        self,
        worker_id: int,
        backend_url: str,
        items: list[tuple[TrialConfig, "asyncio.Future[TrialResult]"]],
    ) -> None:
        """Process trials serially on one backend, resolving futures."""
        for config, fut in items:
            try:
                pinned = self._pin_backend(config, backend_url)
                self._logger.debug(
                    f"Worker {worker_id}: {config.trial_name} -> {backend_url}"
                )
                result = await self._execute_trial_with_retries(pinned)
                fut.set_result(result)
            except Exception as exc:
                fut.set_exception(exc)
