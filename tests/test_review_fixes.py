"""Regression tests for issues found in the pre-launch code review."""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting_tools import BinaryQuestion, ForecastReport, MultipleChoiceQuestion, NumericQuestion  # noqa: E402

import main  # noqa: E402
from forecaster import ops, parsing  # noqa: E402
from forecaster.llm_pool import (  # noqa: E402
    AllModelsFailed,
    LlmPool,
    ModelSpec,
    UsageLedger,
    is_daily_quota_error,
    is_rate_limit_error,
    retry_delay_seconds,
)

PER_MINUTE = (
    'litellm.RateLimitError: GeminiException - {"error": {"code": 429, "message": "You exceeded your '
    'current quota, please check your plan and billing details.", "status": "RESOURCE_EXHAUSTED", '
    '"details": [{"quotaId": "GenerateRequestsPerMinutePerProjectPerModel-FreeTier"}, '
    '{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "7s"}]}}'
)
PER_DAY = PER_MINUTE.replace("PerMinute", "PerDay")


class StatusError(Exception):
    def __init__(self, message, status_code):
        super().__init__(message)
        self.status_code = status_code


# ----------------------------------------------------------------------------- quota classification


def test_per_minute_429_is_not_daily():
    error = StatusError(PER_MINUTE, 429)
    assert is_rate_limit_error(error)
    assert not is_daily_quota_error(error)
    assert retry_delay_seconds(error) == pytest.approx(7.0)


def test_per_day_429_is_daily():
    assert is_daily_quota_error(StatusError(PER_DAY, 429))


def test_per_minute_429_retries_same_model_and_is_not_persisted(monkeypatch):
    pool = LlmPool([ModelSpec(label="a", model="x/a", rpm=10000), ModelSpec(label="b", model="x/b", rpm=10000)])
    calls = []

    async def flaky(spec, prompt):
        calls.append(spec.label)
        if len(calls) == 1:
            raise StatusError(PER_MINUTE.replace('"7s"', '"0s"'), 429)
        return "ok"

    async def no_sleep(_):
        return None

    monkeypatch.setattr(pool, "_call_one", flaky)
    monkeypatch.setattr(asyncio, "sleep", no_sleep)
    text, label = asyncio.run(pool.call("hi"))
    assert (text, label, calls) == ("ok", "a", ["a", "a"])
    assert "a" not in pool.ledger.exhausted


def test_rejected_requests_are_refunded(tmp_path, monkeypatch):
    ledger = UsageLedger(str(tmp_path / "u.json"))
    pool = LlmPool([ModelSpec(label="a", model="x/a", rpm=10000, rpd=5)], ledger)

    class FakeLlm:
        async def invoke(self, prompt):
            raise StatusError("400 bad request", 400)

    monkeypatch.setattr(pool, "_make_llm", lambda spec, timeout: FakeLlm())
    with pytest.raises(AllModelsFailed):
        asyncio.run(pool.call("hi"))
    assert ledger.used("a") == 0


def test_repeated_failures_bench_model_for_run(monkeypatch):
    pool = LlmPool([ModelSpec(label="a", model="x/a", rpm=10000)])

    async def boom(spec, prompt):
        raise ValueError("server error")

    monkeypatch.setattr(pool, "_call_one", boom)
    for _ in range(3):
        with pytest.raises(AllModelsFailed):
            asyncio.run(pool.call("hi"))
    assert pool.available() == []


# ----------------------------------------------------------------------------- parsing robustness


def test_binary_ignores_earlier_probability_when_final_line_differs():
    text = "Base rate: Probability: 30%\nMarket says 45%.\nFinal answer -> Probability (YES): 12%"
    assert parsing.parse_binary_probability(text) == pytest.approx(0.12)


def test_binary_final_line_unreadable_raises_instead_of_using_draft():
    text = "Draft Probability: 30%\nreasoning\nProbability: twenty percent"
    with pytest.raises(parsing.ParseError):
        parsing.parse_binary_probability(text)


def test_binary_range_rejected():
    with pytest.raises(parsing.ParseError):
        parsing.parse_binary_probability("Probability: 20-25%")


def test_mc_option_suffix_does_not_steal_value():
    options = ["0", "1-5", "6-10", "More than 10"]
    text = "Final:\n0: 10%\n1-5: 40%\n6-10: 30%\nMore than 10: 20%"
    assert parsing.parse_multiple_choice(text, options) == pytest.approx(
        {"0": 0.1, "1-5": 0.4, "6-10": 0.3, "More than 10": 0.2}
    )


def test_mc_uses_last_complete_block_not_drafts():
    options = ["A", "B", "C"]
    text = "Initial view:\nA: 60%\nB: 30%\nC: 10%\n\nAfter more thought the evidence shifts.\n\nA: 20%\nB: 50%\nC: 30%"
    assert parsing.parse_multiple_choice(text, options)["A"] == pytest.approx(0.2)


def test_mc_incomplete_final_block_is_not_mixed_with_drafts():
    options = ["A", "B", "C"]
    text = "A: 60%\nB: 30%\nC: 10%\n\nreasoning...\n\nA: 20%\nB: 50%"
    assert parsing.parse_multiple_choice(text, options)["A"] == pytest.approx(0.6)


def test_mc_table_and_special_characters():
    options = ["Yes (before 2027)", "No / never", "C++"]
    text = "| Option | Probability |\n|---|---|\n| Yes (before 2027) | 30% |\n| No / never | 60% |\n| C++ | 10% |"
    result = parsing.parse_multiple_choice(text, options)
    assert result["C++"] == pytest.approx(0.1)


def test_numeric_glued_suffix_rejected_not_truncated():
    levels = [10, 50, 90]
    text = "Percentile 10: 3.5GW\nPercentile 50: 4\nPercentile 90: 5"
    with pytest.raises(parsing.ParseError):
        parsing.parse_percentiles(text, levels)


def test_numeric_scale_matches_unit():
    levels = [10, 50, 90]
    text = "Percentile 10: 3.75B\nPercentile 50: 4 billion\nPercentile 90: 5200 million"
    pairs = parsing.parse_percentiles(text, levels, unit="billion USD")
    assert [v for _, v in pairs] == pytest.approx([3.75, 4.0, 5.2])


def test_numeric_alternative_formats():
    levels = [10, 50, 90]
    text = "10th percentile: 1.2e3\nP50: 1,500\n| 90 | 2000 |"
    pairs = parsing.parse_percentiles(text, levels)
    assert [v for _, v in pairs] == pytest.approx([1200.0, 1500.0, 2000.0])


def test_numeric_space_thousands_rejected():
    with pytest.raises(parsing.ParseError):
        parsing.parse_percentiles("Percentile 10: 1 200 000\nPercentile 50: 2\nPercentile 90: 3", [10, 50, 90])


def test_date_month_names():
    text = "Percentile 10: March 1, 2027\nPercentile 50: 2027-06-01\nPercentile 90: 1 Dec 2027"
    pairs = parsing.parse_date_percentiles(text, [10, 50, 90])
    assert pairs[0][1] < pairs[1][1] < pairs[2][1]


# ----------------------------------------------------------------------------- aggregation fixes


def test_floor_mc_keeps_sum_and_floor():
    result = main.floor_mc({"a": 0.97, "b": 0.02, "c": 0.005, "d": 0.005})
    assert sum(result.values()) == pytest.approx(1.0)
    assert min(result.values()) >= 0.01 - 1e-12


def _numeric_question(open_upper=True, open_lower=False):
    now = datetime.now(timezone.utc)
    return NumericQuestion(
        question_text="How many?",
        id_of_question=7,
        id_of_post=7,
        page_url="https://www.metaculus.com/questions/7/",
        close_time=now + timedelta(days=1),
        upper_bound=100.0,
        lower_bound=0.0,
        open_upper_bound=open_upper,
        open_lower_bound=open_lower,
        unit_of_measure="units",
        zero_point=None,
    )


def test_distribution_all_values_above_open_upper_bound():
    pairs = [(p / 100, 160 + p) for p in main.NUMERIC_PERCENTILES]
    distribution = main.FutureEvalBot._distribution_from_pairs(pairs, _numeric_question())
    assert len(distribution.get_cdf()) == 201


def test_distribution_unit_mismatch_rejected():
    pairs = [(p / 100, 1e6 * (1 + p)) for p in main.NUMERIC_PERCENTILES]
    with pytest.raises(parsing.ParseError):
        main.FutureEvalBot._distribution_from_pairs(pairs, _numeric_question())


# ----------------------------------------------------------------------------- repair path


class RepairPool(LlmPool):
    def __init__(self, forecast_answer, repair_answer):
        super().__init__([ModelSpec(label="f", model="x/f", rpm=10000, roles=("forecast", "research"))])
        self.forecast_answer = forecast_answer
        self.repair_answer = repair_answer

    async def call(self, prompt, *, role="forecast", start=0):
        if role == "research":
            if "Copy the forecaster's FINAL answer" in prompt:
                return self.repair_answer, "lite"
            return "Q1: a\nQ2: b", "lite"
        return self.forecast_answer, "flash"


def test_repair_rescues_malformed_binary_answer():
    pool = RepairPool("I land on about one in five, so twenty percent.", "Probability: 20%")
    bot = main.build_bot(pool, publish=False, quiet=True, samples=2)
    question = BinaryQuestion(question_text="Q?", id_of_question=1, id_of_post=1, page_url="https://www.metaculus.com/questions/1/")
    report = asyncio.run(bot.forecast_questions([question], return_exceptions=False))[0]
    assert report.prediction == pytest.approx(0.2)


def test_mc_sample_floor_avoids_library_rejection():
    options = [f"opt{i}" for i in range(12)]
    answer = "\n".join(f"opt{i}: {89 if i == 0 else 1}%" for i in range(12))
    pool = RepairPool(answer, "NONE")
    bot = main.build_bot(pool, publish=False, quiet=True, samples=2)
    question = MultipleChoiceQuestion(
        question_text="Which?", options=options, id_of_question=2, id_of_post=2, page_url="https://www.metaculus.com/questions/2/"
    )
    report = asyncio.run(bot.forecast_questions([question], return_exceptions=False))[0]
    assert sum(o.probability for o in report.prediction.predicted_options) == pytest.approx(1.0)


# ----------------------------------------------------------------------------- runner: attempt cap, alerts


class FakeClient:
    def __init__(self, questions):
        self.questions = questions

    def get_all_open_questions_from_tournament(self, tournament):
        return self.questions if tournament == "t" else []


def test_failed_question_is_capped_and_alerted_once(tmp_path, monkeypatch):
    question = BinaryQuestion(question_text="Q?", id_of_question=11, id_of_post=11, page_url="https://www.metaculus.com/questions/11/", already_forecasted=False)
    monkeypatch.setattr(main, "MetaculusClient", lambda: FakeClient([question]))
    ledger = UsageLedger(str(tmp_path / "u.json"))
    pool = LlmPool([ModelSpec(label="f", model="x/f", rpm=10000, rpd=100)], ledger)
    bot = main.build_bot(pool, publish=False, quiet=True, samples=5)
    attempts = []

    async def always_fail(questions, return_exceptions=False):
        attempts.append(bot.predictions_per_research_report)
        return [RuntimeError("boom 0.37")]

    monkeypatch.setattr(bot, "forecast_questions", always_fail)
    hard_end = main.time.monotonic() + 3600
    for _ in range(4):
        summary = asyncio.run(main.run_tournaments(bot, ["t"], hard_end=hard_end, max_samples=5, publish=False))
    assert attempts == [5, 2]  # second attempt uses fewer samples, then gives up
    assert "gave up" in summary["skipped"][0][2]
    first_alert = main._format_alert({"failed": [("u", "11", "x")], "skipped": [], "comment_problems": []}, pool)
    second_alert = main._format_alert({"failed": [("u", "11", "x")], "skipped": [], "comment_problems": []}, pool)
    assert first_alert and second_alert is None
    assert "0.37" not in str(summary["failed"])


def test_sanitize_filter_masks_numbers_in_logs(caplog):
    handler = logging.StreamHandler()
    root = logging.getLogger()
    root.addHandler(handler)
    try:
        ops.quiet_forecast_logging()
        record = logging.LogRecord("forecasting_tools.x", logging.WARNING, __file__, 1, "Percentiles: [0.1, 42.5]", (), None)
        for flt in handler.filters:
            assert flt.filter(record)
        assert "42.5" not in record.msg
        info = logging.LogRecord("forecasting_tools.x", logging.INFO, __file__, 1, "Forecast 0.3", (), None)
        assert not all(flt.filter(info) for flt in handler.filters)
    finally:
        root.removeHandler(handler)
