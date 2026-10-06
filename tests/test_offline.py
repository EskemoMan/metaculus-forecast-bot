"""
Offline tests: parsers + the full bot pipeline with a fake LLM pool (no network, no publishing).
Run: python -m pytest -q tests
"""
from __future__ import annotations

import asyncio
import itertools
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from forecasting_tools import (  # noqa: E402
    BinaryQuestion,
    DateQuestion,
    MultipleChoiceQuestion,
    NumericQuestion,
)

import main  # noqa: E402
from forecaster import parsing  # noqa: E402
from forecaster import research as research_mod  # noqa: E402
from forecaster.llm_pool import AllModelsFailed, LlmPool, ModelSpec, UsageLedger  # noqa: E402
from forecaster.ops import sanitize  # noqa: E402


# ----------------------------------------------------------------------------- parsers


def test_binary_last_occurrence_wins():
    text = "Base rate suggests Probability: 30% ... after evidence\nProbability: 42%"
    assert parsing.parse_binary_probability(text) == pytest.approx(0.42)


def test_binary_markdown_and_decimal():
    assert parsing.parse_binary_probability("**Probability: 7.5%**") == pytest.approx(0.075)


def test_binary_missing_raises():
    with pytest.raises(parsing.ParseError):
        parsing.parse_binary_probability("I think it's likely.")


def test_multiple_choice_basic_and_normalized():
    options = ["Trump", "Harris", "Other (incl. none)"]
    text = "reasoning...\nTrump: 50%\nHarris: 45%\nOther (incl. none): 6%"
    result = parsing.parse_multiple_choice(text, options)
    assert sum(result.values()) == pytest.approx(1.0)
    assert result["Trump"] == pytest.approx(50 / 101)


def test_multiple_choice_numeric_option_names():
    options = ["0", "1-2", "3 or more"]
    text = "Final probabilities:\n0: 20%\n1-2: 50%\n3 or more: 30%"
    result = parsing.parse_multiple_choice(text, options)
    assert result == pytest.approx({"0": 0.2, "1-2": 0.5, "3 or more": 0.3})


def test_multiple_choice_missing_option_raises():
    with pytest.raises(parsing.ParseError):
        parsing.parse_multiple_choice("A: 50%\nB: 50%", ["A", "B", "C"])


def test_percentiles_with_commas_and_dollars():
    levels = main.NUMERIC_PERCENTILES
    lines = "\n".join(f"Percentile {p}: ${1000 + 10 * p:,}" for p in levels)
    pairs = parsing.parse_percentiles("analysis\n" + lines, levels)
    assert pairs[0] == (0.02, 1020.0)
    assert pairs[-1] == (0.98, 1980.0)


def test_percentiles_decreasing_raises():
    levels = [10, 50, 90]
    with pytest.raises(parsing.ParseError):
        parsing.parse_percentiles("Percentile 10: 5\nPercentile 50: 4\nPercentile 90: 6", levels)


def test_date_percentiles():
    levels = [10, 50, 90]
    text = "Percentile 10: 2027-01-01\nPercentile 50: 2027-03-01\nPercentile 90: 2027-06-30"
    pairs = parsing.parse_date_percentiles(text, levels)
    assert pairs[1][0] == 0.5
    assert pairs[0][1] < pairs[1][1] < pairs[2][1]


def test_sanitize_masks_numbers_but_keeps_urls():
    text = sanitize("value 0.73 bad at https://www.metaculus.com/questions/123/")
    assert "0.73" not in text
    assert "https://www.metaculus.com/questions/123/" in text


# ----------------------------------------------------------------------------- pool


def test_pool_falls_back_on_daily_quota(monkeypatch):
    pool = LlmPool(
        [
            ModelSpec(label="a", model="x/a", rpm=10000),
            ModelSpec(label="b", model="x/b", rpm=10000),
        ]
    )

    async def fake_call_one(spec, prompt):
        if spec.label == "a":
            raise RuntimeError("429 RESOURCE_EXHAUSTED GenerateRequestsPerDayPerProjectPerModel-FreeTier")
        return "ok"

    monkeypatch.setattr(pool, "_call_one", fake_call_one)
    text, label = asyncio.run(pool.call("hi"))
    assert (text, label) == ("ok", "b")
    assert "a" in pool.stats()["exhausted_today"]
    text, label = asyncio.run(pool.call("hi", start=0))
    assert label == "b"


def test_pool_all_fail(monkeypatch):
    pool = LlmPool([ModelSpec(label="a", model="x/a", rpm=10000)])

    async def boom(spec, prompt):
        raise ValueError("bad request")

    monkeypatch.setattr(pool, "_call_one", boom)
    with pytest.raises(AllModelsFailed):
        asyncio.run(pool.call("hi"))


# ----------------------------------------------------------------------------- full pipeline


class FakePool(LlmPool):
    def __init__(self, answers):
        super().__init__([ModelSpec(label="fake", model="fake/fake", rpm=10000, roles=("forecast", "research"))])
        self._answers = answers
        self._cycle = itertools.count()
        self.prompts = []

    async def call(self, prompt, *, role="forecast", start=0):
        self.prompts.append((role, prompt))
        if role == "research":
            if "Write 2 web search queries" in prompt:
                return "Q1: latest news on x\nQ2: x historical base rate", "fake"
            return "Research brief: status quo, base rates, markets.", "fake"
        index = next(self._cycle)
        return self._answers[index % len(self._answers)], f"fake{index}"


def _common(**kwargs):
    now = datetime.now(timezone.utc)
    base = dict(
        question_text="Will X happen?",
        id_of_question=1,
        id_of_post=1,
        page_url="https://www.metaculus.com/questions/1/",
        background_info="bg",
        resolution_criteria="rc",
        fine_print="fp",
        close_time=now + timedelta(days=1),
    )
    base.update(kwargs)
    return base


def _bot(pool):
    return main.build_bot(pool, publish=False, quiet=True, samples=5)


def test_binary_pipeline_median_and_clip():
    pool = FakePool(["Probability: 10%", "Probability: 20%", "Probability: 30%", "Probability: 99%", "Probability: 25%"])
    question = BinaryQuestion(**_common())
    report = asyncio.run(_bot(pool).forecast_questions([question], return_exceptions=False))[0]
    assert report.prediction == pytest.approx(0.25)
    assert any(role == "research" for role, _ in pool.prompts), "research step should run"


def test_binary_pipeline_tolerates_some_parse_failures():
    pool = FakePool(["Probability: 40%", "no number here", "Probability: 60%", "garbage", "Probability: 50%"])
    question = BinaryQuestion(**_common())
    report = asyncio.run(_bot(pool).forecast_questions([question], return_exceptions=False))[0]
    assert report.prediction == pytest.approx(0.5)


def test_mc_pipeline_mean_and_floor():
    options = ["A", "B", "C"]
    pool = FakePool(["A: 70%\nB: 30%\nC: 0%", "A: 50%\nB: 50%\nC: 0%"])
    question = MultipleChoiceQuestion(**_common(options=options))
    report = asyncio.run(_bot(pool).forecast_questions([question], return_exceptions=False))[0]
    probs = {o.option_name: o.probability for o in report.prediction.predicted_options}
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["C"] > 0
    assert probs["A"] > probs["B"]


def _numeric_answer(shift):
    return "\n".join(f"Percentile {p}: {shift + p}" for p in main.NUMERIC_PERCENTILES)


def test_numeric_pipeline_mean_cdf():
    pool = FakePool([_numeric_answer(0), _numeric_answer(5), _numeric_answer(10)])
    question = NumericQuestion(
        **_common(
            upper_bound=150.0,
            lower_bound=0.0,
            open_upper_bound=True,
            open_lower_bound=False,
            unit_of_measure="units",
            zero_point=None,
        )
    )
    report = asyncio.run(_bot(pool).forecast_questions([question], return_exceptions=False))[0]
    cdf = report.prediction.get_cdf()
    assert len(cdf) == 201
    heights = [p.percentile for p in cdf]
    assert all(b >= a for a, b in zip(heights, heights[1:]))
    assert heights[0] == pytest.approx(0.0, abs=1e-9)  # closed lower bound


def test_numeric_out_of_range_values_are_clamped_not_crashing():
    answer = "\n".join(f"Percentile {p}: {p * 1000}" for p in main.NUMERIC_PERCENTILES)
    pool = FakePool([answer, _numeric_answer(10)])
    question = NumericQuestion(
        **_common(
            upper_bound=100.0,
            lower_bound=0.0,
            open_upper_bound=True,
            open_lower_bound=True,
            unit_of_measure="units",
            zero_point=None,
        )
    )
    report = asyncio.run(_bot(pool).forecast_questions([question], return_exceptions=False))[0]
    assert len(report.prediction.get_cdf()) == 201


def test_date_pipeline():
    start = datetime(2027, 1, 1, tzinfo=timezone.utc)
    lines = "\n".join(
        f"Percentile {p}: {(start + timedelta(days=3 * p)).date().isoformat()}"
        for p in main.NUMERIC_PERCENTILES
    )
    pool = FakePool([lines])
    question = DateQuestion(
        **_common(
            upper_bound=datetime(2028, 1, 1, tzinfo=timezone.utc),
            lower_bound=datetime(2026, 10, 1, tzinfo=timezone.utc),
            open_upper_bound=True,
            open_lower_bound=False,
        )
    )
    report = asyncio.run(_bot(pool).forecast_questions([question], return_exceptions=False))[0]
    assert len(report.prediction.get_cdf()) == 201


# ----------------------------------------------------------------------------- budgets + research


def test_ledger_budget_and_persistence(tmp_path, monkeypatch):
    path = str(tmp_path / "usage.json")
    ledger = UsageLedger(path)
    pool = LlmPool([ModelSpec(label="a", model="x/a", rpm=10000, rpd=2), ModelSpec(label="b", model="x/b", rpm=10000, rpd=1)], ledger)

    async def ok(spec, prompt):
        pool.ledger.add(spec.label)
        return "fine"

    monkeypatch.setattr(pool, "_call_one", ok)
    labels = [asyncio.run(pool.call("hi"))[1] for _ in range(3)]
    assert labels == ["a", "a", "b"]
    assert pool.remaining() == 0
    with pytest.raises(AllModelsFailed):
        asyncio.run(pool.call("hi"))
    ledger.save()
    reloaded = UsageLedger(path)
    assert reloaded.used("a") == 2 and reloaded.used("b") == 1


def test_choose_samples():
    assert main.choose_samples(80, 5) == 5
    assert main.choose_samples(30, 5) == 4
    assert main.choose_samples(15, 5) == 3
    assert main.choose_samples(5, 5) == 2
    assert main.choose_samples(80, 3) == 3


def test_research_uses_search_results(monkeypatch):
    calls = []

    def fake_search(query, *, topic="general", days=None, max_results=6):
        calls.append((query, topic))
        return [{"title": "Headline", "url": "https://example.com/a", "published_date": "2026-10-01", "content": "Something happened."}]

    monkeypatch.setattr(research_mod, "tavily_search", fake_search)
    pool = FakePool(["Probability: 50%"])
    question = BinaryQuestion(**_common())
    text = asyncio.run(research_mod.research_question(pool, question))
    assert [t for _, t in calls] == ["news", "general"]
    assert calls[0][0] == "latest news on x"
    assert "Research brief" in text and "https://example.com/a" in text


def test_research_without_search_key_degrades_gracefully(monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    pool = FakePool(["Probability: 50%"])
    text = asyncio.run(research_mod.research_question(pool, BinaryQuestion(**_common())))
    assert "unavailable" in text.lower()
