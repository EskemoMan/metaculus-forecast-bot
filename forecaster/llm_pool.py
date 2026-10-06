"""
A pool of (mostly free-tier) LLMs with per-model pacing, daily-quota accounting and fallback.

Free tiers have small per-minute and per-day request limits, so:
- each model has a minimum interval between calls (derived from its RPM limit);
- each model has a daily request budget (RPD). Usage is persisted across runs in a small JSON
  ledger (restored/saved via the GitHub Actions cache) and resets at midnight Pacific, which is
  when Google resets free-tier quotas;
- only a 429 that names a PER-DAY quota marks a model exhausted for the day. Gemini uses the
  same "You exceeded your current quota" text for per-minute limits; only the quotaId differs;
- a per-minute 429 waits for the server's suggested retry delay (capped) and retries once; a
  second one benches the model for the rest of THIS run only;
- requests the provider definitely rejected (HTTP 4xx/5xx) are refunded to the daily budget;
- repeated non-quota failures bench a model for the rest of this run (circuit breaker);
- an optional deadline stops trying new models when time is nearly up.
Callers pass `start` to rotate which model is tried first, so the samples of one question are
spread across different models (a cheap ensemble).
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import re
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

from forecasting_tools import GeneralLlm

logger = logging.getLogger(__name__)

USAGE_FILE_DEFAULT = ".usage/usage.json"
FAILURES_BEFORE_BENCH = 3
MAX_RETRY_DELAY_S = 60.0


@dataclass
class ModelSpec:
    label: str
    model: str  # litellm model string, e.g. "gemini/gemini-3.8-flash"
    rpm: float = 5.0  # requests per minute we allow ourselves
    rpd: int | None = None  # requests per (Pacific) day we allow ourselves; None = unlimited
    roles: tuple[str, ...] = ("forecast",)  # "forecast" and/or "research"
    requires: tuple[str, ...] = ()  # env vars that must be set to use this model
    kwargs: dict = field(default_factory=dict)  # extra litellm kwargs for every call
    env_kwargs: dict = field(default_factory=dict)  # litellm kwarg -> env var name (e.g. api_key)
    timeout: float = 240.0


class AllModelsFailed(RuntimeError):
    pass


def pacific_date(now: datetime | None = None) -> str:
    """Date in US Pacific time (Gemini free-tier quotas reset at midnight PT)."""
    now = now or datetime.now(timezone.utc)
    try:
        from zoneinfo import ZoneInfo

        return now.astimezone(ZoneInfo("America/Los_Angeles")).date().isoformat()
    except Exception:  # noqa: BLE001 - tzdata missing (e.g. Windows without tzdata)
        return (now - timedelta(hours=8)).date().isoformat()


# ----------------------------------------------------------------------------- error classification

_DAILY_MARKERS = (
    "perday",
    "per_day",
    "per day",
    "requests per day",
    "bymodelbyday",
    "free-models-per-day",
    "insufficient_quota",
)


def _status_code(error: Exception) -> int | None:
    status = getattr(error, "status_code", None)
    return status if isinstance(status, int) else None


def is_rate_limit_error(error: Exception) -> bool:
    if _status_code(error) == 429 or "ratelimit" in type(error).__name__.lower():
        return True
    head = str(error)[:4000].lower()
    return "resource_exhausted" in head or "too many requests" in head


def is_daily_quota_error(error: Exception) -> bool:
    """Only true when the error names a per-day quota (Gemini: ...PerDayPerProjectPerModel...)."""
    text = str(error)[:6000].lower()
    return any(marker in text for marker in _DAILY_MARKERS)


def retry_delay_seconds(error: Exception) -> float:
    text = str(error)[:6000].lower()
    match = re.search(r"retry in ([\d.]+)\s*s", text) or re.search(r'"retrydelay":\s*"([\d.]+)s"', text)
    if match:
        return min(float(match.group(1)), MAX_RETRY_DELAY_S)
    return 35.0


# ----------------------------------------------------------------------------- ledger


class UsageLedger:
    """
    Per-model request counts, per-question attempt counts and alerted-question ids for the
    current Pacific day, persisted to a JSON file.
    """

    def __init__(self, path: str | None) -> None:
        self.path = path
        self._reset(pacific_date())
        if path and os.path.exists(path):
            try:
                with open(path, encoding="utf-8") as handle:
                    data = json.load(handle)
                if data.get("date") == self.date:
                    self.calls = {k: int(v) for k, v in data.get("calls", {}).items()}
                    self.exhausted = set(data.get("exhausted", []))
                    self.attempts = {k: int(v) for k, v in data.get("attempts", {}).items()}
                    self.alerted = set(data.get("alerted", []))
                self.pending_comments = list(data.get("pending_comments", []))
            except Exception as error:  # noqa: BLE001
                logger.warning(f"Could not read usage ledger: {type(error).__name__}")

    def _reset(self, date: str) -> None:
        self.date = date
        self.calls: dict[str, int] = {}
        self.exhausted: set[str] = set()
        self.attempts: dict[str, int] = {}
        self.alerted: set[str] = set()
        if not hasattr(self, "pending_comments"):
            self.pending_comments: list[int] = []

    def roll_if_new_day(self) -> None:
        today = pacific_date()
        if today != self.date:
            self._reset(today)

    def used(self, label: str) -> int:
        self.roll_if_new_day()
        return self.calls.get(label, 0)

    def add(self, label: str, amount: int = 1) -> None:
        self.roll_if_new_day()
        self.calls[label] = max(0, self.calls.get(label, 0) + amount)
        self.save()

    def save(self) -> None:
        if not self.path:
            return
        directory = os.path.dirname(self.path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "date": self.date,
                    "calls": self.calls,
                    "exhausted": sorted(self.exhausted),
                    "attempts": self.attempts,
                    "alerted": sorted(self.alerted),
                    "pending_comments": self.pending_comments,
                },
                handle,
                indent=2,
            )
        os.replace(tmp, self.path)


# ----------------------------------------------------------------------------- pool


class LlmPool:
    def __init__(self, specs: list[ModelSpec], ledger: UsageLedger | None = None) -> None:
        self.specs = [s for s in specs if all(os.getenv(name) for name in s.requires)]
        self.ledger = ledger or UsageLedger(None)
        self.deadline: float | None = None  # time.monotonic() value; set per question by the runner
        self._locks = {s.label: asyncio.Lock() for s in self.specs}
        self._last_call = {s.label: 0.0 for s in self.specs}
        self._benched: set[str] = set()  # unavailable for the rest of this run
        self.failures_by_model: dict[str, int] = {s.label: 0 for s in self.specs}

    # ------------------------------------------------------------------ info
    def _has_budget(self, spec: ModelSpec) -> bool:
        if spec.label in self._benched or spec.label in self.ledger.exhausted:
            return False
        return spec.rpd is None or self.ledger.used(spec.label) < spec.rpd

    def available(self, role: str = "forecast") -> list[ModelSpec]:
        self.ledger.roll_if_new_day()
        return [s for s in self.specs if role in s.roles and self._has_budget(s)]

    def remaining(self, role: str = "forecast") -> int:
        """Remaining daily calls across models with that role (unlimited models count as 1000)."""
        total = 0
        for spec in self.available(role):
            total += 1000 if spec.rpd is None else spec.rpd - self.ledger.used(spec.label)
        return total

    def stats(self) -> dict:
        return {
            "date_pt": self.ledger.date,
            "configured": [s.label for s in self.specs],
            "exhausted_today": sorted(self.ledger.exhausted),
            "benched_this_run": sorted(self._benched),
            "used_today": dict(self.ledger.calls),
            "failures_this_run": self.failures_by_model,
        }

    def _time_left(self) -> float:
        if self.deadline is None:
            return float("inf")
        return self.deadline - time.monotonic()

    # ------------------------------------------------------------------ calls
    async def _pace(self, spec: ModelSpec) -> None:
        min_interval = 60.0 / max(spec.rpm, 0.1)
        wait = self._last_call[spec.label] + min_interval - time.monotonic()
        if wait > 0:
            await asyncio.sleep(wait + random.uniform(0, 0.5))
        self._last_call[spec.label] = time.monotonic()

    def _make_llm(self, spec: ModelSpec, timeout: float) -> GeneralLlm:
        kwargs = dict(spec.kwargs)
        for kwarg_name, env_name in spec.env_kwargs.items():
            kwargs[kwarg_name] = os.getenv(env_name)
        return GeneralLlm(
            model=spec.model,
            allowed_tries=1,
            timeout=timeout,
            populate_citations=False,
            **kwargs,
        )

    async def _call_one(self, spec: ModelSpec, prompt: str) -> str:
        async with self._locks[spec.label]:
            await self._pace(spec)
            timeout = max(30.0, min(spec.timeout, self._time_left() - 10))
            self.ledger.add(spec.label)  # reserve before calling so concurrent samples can't overshoot
            try:
                return await self._make_llm(spec, timeout).invoke(prompt)
            except Exception as error:
                status = _status_code(error)
                if status is not None and 400 <= status < 600 and status != 408:
                    self.ledger.add(spec.label, -1)  # provider rejected it: not served, refund
                raise

    async def call(self, prompt: str, *, role: str = "forecast", start: int = 0) -> tuple[str, str]:
        """
        Returns (text, model_label). Tries models with the given role in rotated order
        starting at `start`.
        """
        candidates = self.available(role)
        if not candidates:
            raise AllModelsFailed(f"No '{role}' models with remaining quota")
        offset = start % len(candidates)
        ordered = candidates[offset:] + candidates[:offset]
        errors: list[str] = []
        for spec in ordered:
            if not self._has_budget(spec):
                continue
            if self._time_left() < 45:
                errors.append("deadline reached")
                break
            for attempt in range(2):
                try:
                    text = await self._call_one(spec, prompt)
                    if not text or not text.strip():
                        raise RuntimeError("empty response")
                    return text, spec.label
                except Exception as error:  # noqa: BLE001 - provider errors vary widely
                    self.failures_by_model[spec.label] += 1
                    name = type(error).__name__
                    if is_rate_limit_error(error) and is_daily_quota_error(error):
                        logger.warning(f"[pool] {spec.label}: daily quota exhausted")
                        self.ledger.exhausted.add(spec.label)
                        self.ledger.save()
                        errors.append(f"{spec.label}: daily quota")
                        break
                    status = _status_code(error)
                    if status in (500, 502, 503, 504) and attempt == 0 and self._time_left() > 90:
                        delay = random.uniform(8, 15)
                        logger.warning(f"[pool] {spec.label}: server error {status}, retrying in {delay:.0f}s")
                        await asyncio.sleep(delay)
                        continue
                    if is_rate_limit_error(error):
                        if attempt == 0 and self._time_left() > retry_delay_seconds(error) + 60:
                            delay = retry_delay_seconds(error) + random.uniform(1, 4)
                            logger.warning(f"[pool] {spec.label}: per-minute limit, waiting {delay:.0f}s")
                            await asyncio.sleep(delay)
                            continue
                        self._benched.add(spec.label)
                        errors.append(f"{spec.label}: rate limited")
                        break
                    logger.warning(f"[pool] {spec.label}: {name} (status {_status_code(error)})")
                    errors.append(f"{spec.label}: {name}")
                    if self.failures_by_model[spec.label] >= FAILURES_BEFORE_BENCH:
                        self._benched.add(spec.label)
                    break
        raise AllModelsFailed("All models failed: " + "; ".join(errors))


def load_pool_from_env(default_specs: list[ModelSpec]) -> LlmPool:
    """
    BOT_MODEL_POOL (optional) may hold a JSON list of ModelSpec dicts to override the defaults
    without a code change (e.g. once Metaculus credits arrive).
    BOT_USAGE_FILE sets where the daily usage ledger lives (default .usage/usage.json).
    """
    ledger = UsageLedger(os.getenv("BOT_USAGE_FILE") or USAGE_FILE_DEFAULT)
    raw = os.getenv("BOT_MODEL_POOL")
    if raw:
        specs = []
        for item in json.loads(raw):
            item["requires"] = tuple(item.get("requires", ()))
            item["roles"] = tuple(item.get("roles", ("forecast",)))
            specs.append(ModelSpec(**item))
        return LlmPool(specs, ledger)
    return LlmPool(default_specs, ledger)
