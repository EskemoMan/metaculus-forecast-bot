"""
Operational helpers: log hygiene and GitHub-issue alerts.

Rule compliance + competitive hygiene: this repo is public, so GitHub Actions
logs are public. In tournament mode we never log forecast values or reasoning
for open questions (other bots could copy them, and the tournament forbids the
bot maker from previewing open-question forecasts and then tweaking the bot).
Alerts contain only counts, question URLs, model names and sanitized error
types.
"""
from __future__ import annotations

import logging
import os
import re

import requests

logger = logging.getLogger(__name__)

ALERT_LABEL = "bot-alert"
ALERT_TITLE = "Forecast bot needs attention"


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def sanitize(message: str, limit: int = 180) -> str:
    """Strip numbers (could be forecast values) and long text from error messages."""
    text = message.replace("\n", " ")
    # keep URLs (question links) but mask other numbers
    parts = re.split(r"(https?://\S+)", text)
    parts = [p if p.startswith("http") else _NUMBER_RE.sub("#", p) for p in parts]
    text = "".join(parts)
    return text[:limit] + ("..." if len(text) > limit else "")


class _SanitizeFilter(logging.Filter):
    """Reduces every log record to a short, number-masked one-liner without tracebacks."""

    def filter(self, record: logging.LogRecord) -> bool:
        if record.levelno < logging.WARNING:
            return False
        try:
            message = record.getMessage()
        except Exception:  # noqa: BLE001
            message = str(record.msg)
        record.msg = sanitize(message, 200)
        record.args = ()
        record.exc_info = None
        record.exc_text = None
        record.stack_info = None
        return True


def quiet_forecast_logging() -> None:
    """
    Tournament mode: the repo and its Actions logs are public. Library warnings/errors can embed
    forecast values (e.g. failed-sample exceptions, distribution validation errors, comment
    markdown), so every handler gets a filter that drops sub-WARNING records and sanitizes the rest.
    The run summary is printed separately with print().
    """
    root = logging.getLogger()
    for handler in root.handlers:
        handler.addFilter(_SanitizeFilter())
    for name in ("LiteLLM", "litellm", "httpx"):
        logging.getLogger(name).setLevel(logging.WARNING)


def _github_context() -> tuple[str, str] | None:
    token = os.getenv("GITHUB_TOKEN")
    repo = os.getenv("GITHUB_REPOSITORY")
    if not token or not repo:
        return None
    return token, repo


def post_alert(body: str) -> None:
    """Create (or comment on) a single open 'bot-alert' issue in this repo."""
    context = _github_context()
    if context is None:
        logger.warning("No GITHUB_TOKEN/GITHUB_REPOSITORY; alert not posted:\n%s", body)
        return
    token, repo = context
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "application/vnd.github+json",
        "X-GitHub-Api-Version": "2022-11-28",
    }
    run_url = ""
    if os.getenv("GITHUB_RUN_ID"):
        run_url = f"\n\nRun: https://github.com/{repo}/actions/runs/{os.getenv('GITHUB_RUN_ID')}"
    try:
        existing = requests.get(
            f"https://api.github.com/repos/{repo}/issues",
            headers=headers,
            params={"state": "open", "labels": ALERT_LABEL, "per_page": 5},
            timeout=30,
        )
        existing.raise_for_status()
        issues = existing.json()
        if issues:
            number = issues[0]["number"]
            requests.post(
                f"https://api.github.com/repos/{repo}/issues/{number}/comments",
                headers=headers,
                json={"body": body + run_url},
                timeout=30,
            ).raise_for_status()
        else:
            requests.post(
                f"https://api.github.com/repos/{repo}/issues",
                headers=headers,
                json={"title": ALERT_TITLE, "body": body + run_url, "labels": [ALERT_LABEL]},
                timeout=30,
            ).raise_for_status()
    except Exception as error:  # noqa: BLE001
        logger.warning(f"Could not post GitHub alert: {type(error).__name__}")
