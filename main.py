"""
Metaculus FutureEval forecasting bot (built on Metaculus' metac-bot-template / forecasting-tools).

Design (see README.md for the full write-up):
- Research: 2 Tavily web searches (recent news + base rates) summarised by a cheap Flash-Lite
  model into a brief; raw results kept for verification. Optional AskNews news.
- Forecast: up to N independent samples spread across the free Gemini Flash models (a cheap
  ensemble), parsed with regexes; a Flash-Lite "repair" call rescues malformed final answers.
- Aggregate: binary = median (clipped); multiple choice = mean (floored at 1%, renormalised);
  numeric/discrete/date = point-wise mean of CDFs (a mixture; never worse than the average
  member under log scoring). Conditional questions use the library default (yes/no branches).
- Ops: each run forecasts only questions not yet forecast (never re-forecasts in the
  tournament), never-attempted questions first, soonest-closing first, within a wall-clock
  budget; at most 2 attempts per question per day; never logs forecast values in tournament
  mode (public repo); opens/updates a GitHub issue for new problems.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import re
import statistics
import sys
import time
from datetime import datetime, timezone
from typing import Callable, Literal

import dotenv

from bot_helpers import silence_noisy_dependencies

silence_noisy_dependencies()

from forecasting_tools import (  # noqa: E402
    AskNewsSearcher,
    BinaryQuestion,
    ConditionalQuestion,
    DateQuestion,
    ForecastBot,
    ForecastReport,
    MetaculusClient,
    MetaculusQuestion,
    MultipleChoiceQuestion,
    NumericDistribution,
    NumericQuestion,
    Percentile,
    PredictedOptionList,
    PredictionTypes,
    ReasonedPrediction,
    clean_indents,
)
from forecasting_tools.data_models.multiple_choice_report import PredictedOption  # noqa: E402

from forecaster import parsing  # noqa: E402
from forecaster.llm_pool import AllModelsFailed, LlmPool, ModelSpec, load_pool_from_env  # noqa: E402
from forecaster.ops import post_alert, quiet_forecast_logging, sanitize  # noqa: E402
from forecaster.research import demote_headings, research_question, tavily_search  # noqa: E402

dotenv.load_dotenv()
logger = logging.getLogger(__name__)
PROCESS_START = time.monotonic()

FALL_2026_TOURNAMENT_ID = 33121  # https://www.metaculus.com/tournament/fall-futureeval-2026/
MINIBENCH_ID = "minibench"  # rotates bi-weekly; the slug always points at the current round
BOT_TESTING_AREA = "bot-testing-area"

NUMERIC_PERCENTILES = [2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 98]
BINARY_CLIP = (0.015, 0.985)
MC_FLOOR = 0.01  # the library's PredictedOptionList clamps options to [0.01, 0.99]
MAX_ATTEMPTS_PER_DAY = 2
PER_QUESTION_CAP_S = 15 * 60

# Gemini free tier (Oct 2026): Flash models ~5 RPM / ~20 requests per day EACH (per project);
# Flash-Lite ~500/day. No free search grounding on Gemini 3.x, so web search uses Tavily.
# Budgets below keep a small safety margin. Unavailable models (missing env var) are skipped.
# Override without a code change via the BOT_MODEL_POOL repository variable (JSON).
_GEMINI = ("GEMINI_API_KEY",)
_HIGH = {"reasoning_effort": "high"}
DEFAULT_MODEL_SPECS: list[ModelSpec] = [
    # --- forecasters (strongest first) ---
    ModelSpec(label="gemini-3.8-flash", model="gemini/gemini-3.8-flash", rpm=4, rpd=19, requires=_GEMINI, kwargs=_HIGH),
    ModelSpec(label="gemini-3.7-flash", model="gemini/gemini-3.7-flash", rpm=4, rpd=19, requires=_GEMINI, kwargs=_HIGH),
    ModelSpec(label="gemini-3.6-flash", model="gemini/gemini-3.6-flash", rpm=4, rpd=19, requires=_GEMINI, kwargs=_HIGH),
    ModelSpec(label="gemini-3.5-flash", model="gemini/gemini-3.5-flash", rpm=4, rpd=19, requires=_GEMINI, kwargs=_HIGH),
    ModelSpec(label="gemini-3-flash-preview", model="gemini/gemini-3-flash-preview", rpm=4, rpd=18, requires=_GEMINI),
    # Optional free fallback (only used if its key is configured).
    ModelSpec(label="groq-qwen3.8-27b", model="groq/qwen/qwen3.8-27b", rpm=10, rpd=900, requires=("GROQ_API_KEY",)),
    # --- research / answer-repair helpers (cheap, high daily quota) ---
    ModelSpec(label="gemini-3.5-flash-lite", model="gemini/gemini-3.5-flash-lite", rpm=8, rpd=450, roles=("research",), requires=_GEMINI),
    ModelSpec(label="gemini-3.1-flash-lite", model="gemini/gemini-3.1-flash-lite", rpm=8, rpd=450, roles=("research",), requires=_GEMINI),
]


def choose_samples(remaining: int, maximum: int) -> int:
    """
    Samples per question from the remaining daily forecaster quota (in samples). Strong samples
    beat padding with weak models, so we shrink N rather than fall back to sub-Flash models.
    """
    if remaining >= 45:
        n = maximum
    elif remaining >= 25:
        n = min(maximum, 4)
    elif remaining >= 12:
        n = min(maximum, 3)
    else:
        n = min(maximum, 2)
    return max(1, min(n, remaining))


def floor_mc(probs: dict[str, float], floor: float = MC_FLOOR) -> dict[str, float]:
    """Raise options below `floor` to the floor and rescale the others so the total stays 1."""
    total = sum(probs.values())
    current = {k: v / total for k, v in probs.items()}
    fixed: set[str] = set()
    while True:
        free = [k for k in current if k not in fixed]
        mass = 1.0 - floor * len(fixed)
        free_total = sum(current[k] for k in free) or 1.0
        updated = {k: (floor if k in fixed else current[k] * mass / free_total) for k in current}
        low = [k for k in free if updated[k] < floor]
        if not low:
            return updated
        fixed |= set(low)
        current = updated


def calls_per_sample(question: MetaculusQuestion) -> int:
    return 2 if isinstance(question, ConditionalQuestion) else 1


class FutureEvalBot(ForecastBot):
    """Forecasting bot for the Metaculus Fall 2026 FutureEval tournament + MiniBench."""

    _max_concurrent_questions = 1
    _concurrency_limiter = asyncio.Semaphore(_max_concurrent_questions)

    def __init__(self, *args, pool: LlmPool, quiet: bool = False, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.pool = pool
        self.quiet = quiet
        self._sample_counters: dict[str, int] = {}

    def _next_sample_index(self, question: MetaculusQuestion) -> int:
        key = str(question.id_of_question or question.page_url)
        index = self._sample_counters.get(key, 0)
        self._sample_counters[key] = index + 1
        return index

    def _log(self, message: str) -> None:
        if not self.quiet:
            logger.info(message)

    ################################### RESEARCH ###################################

    async def run_research(self, question: MetaculusQuestion) -> str:
        async with self._concurrency_limiter:
            sections = [await research_question(self.pool, question)]
            if os.getenv("ASKNEWS_CLIENT_ID") and os.getenv("ASKNEWS_SECRET"):
                try:
                    news = await AskNewsSearcher().call_preconfigured_version(
                        "asknews/news-summaries", question.question_text
                    )
                    sections.append(f"#### Recent news (AskNews)\n{demote_headings(news)}")
                except Exception as error:  # noqa: BLE001
                    logger.warning(f"AskNews failed: {type(error).__name__}")
            research = "\n\n".join(sections)
            self._log(f"Research for {question.page_url}:\n{research}")
            return research

    ################################### SHARED PARTS ###################################

    @staticmethod
    def _question_block(question: MetaculusQuestion, research: str) -> str:
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        return clean_indents(
            f"""
            Question: {question.question_text}

            Background: {question.background_info}

            Resolution criteria (not yet satisfied): {question.resolution_criteria}

            Fine print: {question.fine_print}

            Research (gathered today; check that claims are consistent before relying on them):
            {research}

            Today is {today}.
            """
        )

    def _conditional_note(self, question: MetaculusQuestion) -> str:
        if question.conditional_type not in ["yes", "no"]:
            return ""
        return (
            "This is a conditional question: forecast ONLY the child question given the stated "
            "resolution of the parent question. Never re-forecast the parent question."
        )

    async def _sample(self, question: MetaculusQuestion, prompt: str) -> tuple[str, str]:
        start = self._next_sample_index(question)
        return await self.pool.call(prompt, role="forecast", start=start)

    async def _parse_or_repair(self, reasoning: str, parse: Callable[[str], object], answer_format: str) -> object:
        """Parse the final answer; if that fails, ask a cheap model to restate it in the exact format."""
        try:
            return parse(reasoning)
        except parsing.ParseError:
            pass
        tail = reasoning[-6000:]
        prompt = clean_indents(
            f"""
            Below is the end of a forecaster's analysis. Copy the forecaster's FINAL answer into
            exactly the format shown, using the same numbers. Do not change, average or invent
            any value. If no final answer is given, reply exactly: NONE

            FORMAT:
            {answer_format}

            ANALYSIS (end):
            {tail}
            """
        )
        try:
            repaired, _ = await self.pool.call(prompt, role="research")
        except AllModelsFailed as error:
            raise parsing.ParseError("unparseable answer and repair unavailable") from error
        if repaired.strip().upper().startswith("NONE"):
            raise parsing.ParseError("no final answer in sample")
        return parse(repaired)

    @staticmethod
    def _reasoned(model: str, reasoning: str, value) -> ReasonedPrediction:
        return ReasonedPrediction(prediction_value=value, reasoning=f"[{model}]\n{demote_headings(reasoning)}")

    ################################### BINARY ###################################

    async def _run_forecast_on_binary(self, question: BinaryQuestion, research: str) -> ReasonedPrediction[float]:
        prompt = clean_indents(
            f"""
            You are an expert superforecaster in a tournament scored by log score against other
            forecasters. Overconfidence on a wrong answer is punished hard; so is needless hedging.

            {self._question_block(question, research)}

            Reason briefly through:
            (a) Time left until resolution and exactly what must happen for YES.
            (b) The status-quo outcome if nothing changes (the world usually changes slowly).
            (c) A base rate from the best reference class, and the probability it implies.
            (d) How the latest evidence should move you from the base rate. Note any key claim
                that looks stale, unsourced or contradictory and discount it.
            (e) Market or expert signals, if any, and how much weight they deserve.
            (f) The strongest case for NO and the strongest case for YES.
            {self._conditional_note(question)}

            The very last line must be exactly: "Probability: ZZ%" with ZZ a single number between 1 and 99.
            """
        )
        reasoning, model = await self._sample(question, prompt)
        probability = await self._parse_or_repair(reasoning, parsing.parse_binary_probability, "Probability: ZZ%")
        probability = min(max(float(probability), 0.01), 0.99)
        self._log(f"Binary sample for {question.page_url} via {model}: {probability}")
        return self._reasoned(model, reasoning, probability)

    ################################### MULTIPLE CHOICE ###################################

    async def _run_forecast_on_multiple_choice(
        self, question: MultipleChoiceQuestion, research: str
    ) -> ReasonedPrediction[PredictedOptionList]:
        options_text = "\n".join(f"{option}: XX%" for option in question.options)
        prompt = clean_indents(
            f"""
            You are an expert superforecaster in a tournament scored by log score against other
            forecasters.

            {self._question_block(question, research)}

            The options are: {question.options}

            Reason briefly through:
            (a) Time left and the status-quo outcome if nothing changes.
            (b) Base rates or reference classes for each plausible option.
            (c) How the latest evidence and any market or expert signals move you.
            (d) A scenario producing an unexpected outcome. Give every option at least 1%.
            {self._conditional_note(question)}

            End with your final probabilities, one per line, with each option name written exactly
            as given, in this order, summing to 100%:
            {options_text}
            """
        )
        reasoning, model = await self._sample(question, prompt)
        options = list(question.options)
        probabilities = await self._parse_or_repair(
            reasoning, lambda text: parsing.parse_multiple_choice(text, options), options_text
        )
        probabilities = floor_mc(dict(probabilities))  # type: ignore[arg-type]
        prediction = PredictedOptionList(
            predicted_options=[PredictedOption(option_name=name, probability=probabilities[name]) for name in options]
        )
        self._log(f"MC sample for {question.page_url} via {model}: {probabilities}")
        return self._reasoned(model, reasoning, prediction)

    ################################### NUMERIC / DISCRETE / DATE ###################################

    def _bound_messages(self, question: NumericQuestion | DateQuestion) -> tuple[str, str]:
        if isinstance(question, DateQuestion):
            upper = question.upper_bound.date().isoformat()
            lower = question.lower_bound.date().isoformat()
            unit = ""
        else:
            upper = question.nominal_upper_bound if question.nominal_upper_bound is not None else question.upper_bound
            lower = question.nominal_lower_bound if question.nominal_lower_bound is not None else question.lower_bound
            unit = question.unit_of_measure or ""
        if question.open_upper_bound:
            upper_message = f"The question creator thinks the value is likely not higher than {upper} {unit}."
        else:
            upper_message = f"The value cannot be higher than {upper} {unit}."
        if question.open_lower_bound:
            lower_message = f"The question creator thinks the value is likely not lower than {lower} {unit}."
        else:
            lower_message = f"The value cannot be lower than {lower} {unit}."
        return upper_message, lower_message

    @staticmethod
    def _percentile_template(is_date: bool) -> str:
        placeholder = "YYYY-MM-DD" if is_date else "VALUE"
        return "\n".join(f"Percentile {p}: {placeholder}" for p in NUMERIC_PERCENTILES)

    async def _run_forecast_on_numeric(
        self, question: NumericQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_message, lower_message = self._bound_messages(question)
        unit = question.unit_of_measure or "not stated (infer it from the question)"
        prompt = clean_indents(
            f"""
            You are an expert superforecaster in a tournament scored by log score of your full
            probability distribution against other forecasters.

            {self._question_block(question, research)}

            Units for the answer: {unit}
            {lower_message}
            {upper_message}

            Reason briefly through:
            (a) Time left until the value is known.
            (b) The value if nothing changes, and the value if the current trend continues.
            (c) Historical variability: how much has this quantity moved over similar horizons?
            (d) Expert, official or market expectations.
            (e) Unexpected low and high scenarios.
            {self._conditional_note(question)}
            Good forecasters are humble: make the 2nd-98th percentile range wide enough to cover
            surprises, but centre the distribution on the most likely values.

            Formatting rules: write each value as a bare number already expressed in the units
            above. Never append million/billion/thousand/M/B/k, currency symbols or ranges, and
            never use scientific notation. Values must increase from one line to the next.
            End with exactly these lines:
            {self._percentile_template(False)}
            """
        )
        reasoning, model = await self._sample(question, prompt)
        unit_text = question.unit_of_measure
        pairs = await self._parse_or_repair(
            reasoning,
            lambda text: parsing.parse_percentiles(text, NUMERIC_PERCENTILES, unit_text),
            self._percentile_template(False),
        )
        distribution = self._distribution_from_pairs(pairs, question)  # type: ignore[arg-type]
        self._log(f"Numeric sample for {question.page_url} via {model} parsed")
        return self._reasoned(model, reasoning, distribution)

    async def _run_forecast_on_date(
        self, question: DateQuestion, research: str
    ) -> ReasonedPrediction[NumericDistribution]:
        upper_message, lower_message = self._bound_messages(question)
        prompt = clean_indents(
            f"""
            You are an expert superforecaster in a tournament scored by log score of your full
            probability distribution against other forecasters.

            {self._question_block(question, research)}

            {lower_message}
            {upper_message}

            Reason briefly through:
            (a) The date implied by the status quo and by current trends or schedules.
            (b) Historical delays or accelerations for similar events.
            (c) Expert, official or market expectations.
            (d) Unexpectedly early and late scenarios.
            {self._conditional_note(question)}
            Make the 2nd-98th percentile range wide enough to cover surprises.

            Write dates as YYYY-MM-DD, in chronological order. End with exactly these lines:
            {self._percentile_template(True)}
            """
        )
        reasoning, model = await self._sample(question, prompt)
        pairs = await self._parse_or_repair(
            reasoning,
            lambda text: parsing.parse_date_percentiles(text, NUMERIC_PERCENTILES),
            self._percentile_template(True),
        )
        distribution = self._distribution_from_pairs(pairs, question)  # type: ignore[arg-type]
        self._log(f"Date sample for {question.page_url} via {model} parsed")
        return self._reasoned(model, reasoning, distribution)

    @staticmethod
    def _distribution_from_pairs(
        pairs: list[tuple[float, float]], question: NumericQuestion | DateQuestion
    ) -> NumericDistribution:
        if isinstance(question, DateQuestion):
            lower, upper = question.lower_bound.timestamp(), question.upper_bound.timestamp()
        else:
            lower, upper = float(question.lower_bound), float(question.upper_bound)
        span = upper - lower
        raw_values = [value for _, value in pairs]
        # A distribution sitting mostly FAR outside the question's range (10x its width) almost
        # always means a unit/scale mistake (e.g. millions vs units): reject the sample.
        near = [v for v in raw_values if lower - 10 * span <= v <= upper + 10 * span]
        if len(near) < len(raw_values) / 2:
            raise parsing.ParseError("values mostly far outside the question range (unit mismatch?)")
        # The library requires every value within lower-2*span..upper+2*span and at least one
        # within +/-25% of the range.
        values = [min(max(v, lower - 1.5 * span), upper + 1.5 * span) for v in raw_values]
        band_low, band_high = lower - 0.2 * span, upper + 0.2 * span
        if not any(band_low <= v <= band_high for v in values):
            if values[0] > band_high:
                values[0] = band_high
            elif values[-1] < band_low:
                values[-1] = band_low
        percentiles: list[Percentile] = []
        previous = None
        for (level, _), value in zip(pairs, values):
            if previous is not None and value <= previous:
                value = previous + max(abs(span) * 1e-6, 1e-9)
            percentiles.append(Percentile(percentile=level, value=value))
            previous = value
        return NumericDistribution.from_question(percentiles, question)

    ################################### AGGREGATION ###################################

    async def _aggregate_predictions(self, predictions: list[PredictionTypes], question: MetaculusQuestion) -> PredictionTypes:
        if not predictions:
            raise ValueError("Cannot aggregate empty list of predictions")
        if isinstance(question, BinaryQuestion):
            median = float(statistics.median(predictions))  # type: ignore[arg-type]
            return min(max(median, BINARY_CLIP[0]), BINARY_CLIP[1])  # type: ignore[return-value]
        if isinstance(question, MultipleChoiceQuestion):
            names = list(question.options)
            sums = {name: 0.0 for name in names}
            for option_list in predictions:  # type: ignore[assignment]
                by_name = {o.option_name: o.probability for o in option_list.predicted_options}
                for name in names:
                    sums[name] += by_name[name]
            means = floor_mc({name: total / len(predictions) for name, total in sums.items()})
            return PredictedOptionList(  # type: ignore[return-value]
                predicted_options=[PredictedOption(option_name=name, probability=means[name]) for name in names]
            )
        if isinstance(question, (NumericQuestion, DateQuestion)):
            cdfs = [prediction.get_cdf() for prediction in predictions]  # type: ignore[union-attr]
            x_axis = [p.value for p in cdfs[0]]
            mean_heights = [sum(cdf[i].percentile for cdf in cdfs) / len(cdfs) for i in range(len(x_axis))]
            mean_cdf = [Percentile(value=value, percentile=height) for value, height in zip(x_axis, mean_heights)]
            return NumericDistribution.from_question(mean_cdf, question)  # type: ignore[return-value]
        return await super()._aggregate_predictions(predictions, question)


################################### RUNNER ###################################


def _close_time_key(question: MetaculusQuestion) -> float:
    return float("inf") if question.close_time is None else question.close_time.timestamp()


def failure_causes(error: BaseException | None, limit: int = 4) -> str:
    """Leaf exception types + sanitized messages (numbers masked) for a failed question."""
    leaves: list[BaseException] = []

    def walk(exc: BaseException | None, depth: int = 0) -> None:
        if exc is None or depth > 8:
            return
        subs = getattr(exc, "exceptions", None)
        if subs:
            for sub in subs:
                walk(sub, depth + 1)
        elif exc.__cause__ is not None:
            walk(exc.__cause__, depth + 1)
        else:
            leaves.append(exc)

    walk(error)
    counts: dict[str, int] = {}
    for leaf in leaves:
        key = f"{type(leaf).__name__}: {sanitize(str(leaf), 90)}"
        counts[key] = counts.get(key, 0) + 1
    ranked = sorted(counts.items(), key=lambda item: -item[1])[:limit]
    return "; ".join(f"{key} (x{n})" for key, n in ranked) or "unknown"


def _question_key(question: MetaculusQuestion) -> str:
    return str(question.id_of_question or question.id_of_post or question.page_url)


async def _publish(report: ForecastReport, client: MetaculusClient, ledger) -> None:
    """
    Publish forecast + private comment. If the forecast went through but the comment failed,
    remember the post id (no content) so a later run can attach a short comment.
    """
    try:
        await report.publish_report_to_metaculus(metaculus_client=client)
        return
    except Exception as error:
        post_id = report.question.id_of_post
        posted = False
        if post_id is not None:
            try:
                refreshed = client.get_question_by_post_id(post_id)
                posted = bool(getattr(refreshed, "already_forecasted", False))
            except Exception:  # noqa: BLE001
                posted = False
        if posted and post_id is not None:
            if post_id not in ledger.pending_comments:
                ledger.pending_comments.append(post_id)
            ledger.save()
            return
        raise error


def _retry_pending_comments(client: MetaculusClient, ledger, bot_name: str) -> list[str]:
    problems = []
    for post_id in list(ledger.pending_comments):
        text = (
            f"Automated forecast by {bot_name}. The full private reasoning comment failed to post "
            "at forecast time because of an API error; this note confirms the forecast was produced "
            "automatically by the bot (research via web search, ensemble of LLM samples)."
        )
        try:
            client.post_question_comment(post_id, text)
            ledger.pending_comments.remove(post_id)
        except Exception as error:  # noqa: BLE001
            problems.append(f"comment for post {post_id}: {type(error).__name__}")
    ledger.save()
    return problems


async def run_tournaments(
    bot: FutureEvalBot,
    tournaments: list[int | str],
    hard_end: float,
    max_samples: int,
    publish: bool,
) -> dict:
    client = MetaculusClient()
    ledger = bot.pool.ledger
    summary: dict = {"forecast": [], "failed": [], "skipped": [], "open": 0, "comment_problems": []}
    if publish:
        summary["comment_problems"] = _retry_pending_comments(client, ledger, type(bot).__name__)
    for tournament in tournaments:
        try:
            questions = client.get_all_open_questions_from_tournament(tournament)
        except Exception as error:  # noqa: BLE001
            summary["failed"].append((f"tournament {tournament}", "", sanitize(f"{type(error).__name__}: {error}")))
            continue
        pending = [q for q in questions if not q.already_forecasted]
        fresh = sorted((q for q in pending if ledger.attempts.get(_question_key(q), 0) == 0), key=_close_time_key)
        retries = sorted((q for q in pending if ledger.attempts.get(_question_key(q), 0) > 0), key=_close_time_key)
        summary["open"] += len(questions)
        print(f"[{tournament}] open={len(questions)} to_forecast={len(pending)} (retries={len(retries)})")
        for question in fresh + retries:
            key = _question_key(question)
            attempts = ledger.attempts.get(key, 0)
            if attempts >= MAX_ATTEMPTS_PER_DAY:
                summary["skipped"].append((question.page_url, key, f"gave up after {attempts} failed attempts today"))
                continue
            time_left = hard_end - time.monotonic()
            if time_left < 6 * 60:
                summary["skipped"].append((question.page_url, key, "run time budget used up (next run continues)"))
                continue
            cost = calls_per_sample(question)
            affordable = bot.pool.remaining("forecast") // cost
            if affordable < 1:
                summary["skipped"].append((question.page_url, key, "LLM daily quota exhausted"))
                continue
            samples = choose_samples(affordable, max_samples)
            if attempts > 0:
                samples = min(samples, 2)
            bot.predictions_per_research_report = samples
            bot.pool.deadline = time.monotonic() + min(PER_QUESTION_CAP_S, time_left - 4 * 60)
            ledger.attempts[key] = attempts + 1
            ledger.save()
            reports = await bot.forecast_questions([question], return_exceptions=True)
            result = reports[0] if reports else None
            if isinstance(result, ForecastReport) and publish:
                try:
                    await _publish(result, client, ledger)
                except Exception as error:  # noqa: BLE001
                    result = error
            ledger.save()
            if isinstance(result, ForecastReport):
                summary["forecast"].append(question.page_url)
                print(f"  forecast ok ({samples} samples): {question.page_url}")
            else:
                reason = f"{type(result).__name__}; causes: {failure_causes(result)}"
                summary["failed"].append((question.page_url, key, reason))
                print(f"  FAILED: {question.page_url} ({reason})")
    bot.pool.deadline = None
    return summary


def _format_alert(summary: dict, pool: LlmPool) -> str | None:
    """Only report problems not already reported today (avoids an alert every 10 minutes)."""
    ledger = pool.ledger
    new_failed = [f for f in summary["failed"] if f[1] not in ledger.alerted]
    new_skipped = [s for s in summary["skipped"] if s[1] not in ledger.alerted]
    if not (new_failed or new_skipped or summary["comment_problems"]):
        return None
    lines = ["The forecasting bot hit problems (no forecast values are shown here)."]
    if new_failed:
        lines.append("\n**Failed (retried once more later today if still open):**")
        lines += [f"- {url}: {reason}" for url, _, reason in new_failed]
    if new_skipped:
        lines.append("\n**Skipped:**")
        lines += [f"- {url}: {reason}" for url, _, reason in new_skipped]
    if summary["comment_problems"]:
        lines.append("\n**Comment retries failing:**")
        lines += [f"- {p}" for p in summary["comment_problems"]]
    stats = pool.stats()
    lines.append(f"\nModels configured: {', '.join(stats['configured']) or 'none'}")
    if stats["exhausted_today"]:
        lines.append(f"Daily quota exhausted: {', '.join(stats['exhausted_today'])}")
    for item in new_failed + new_skipped:
        if item[1]:
            ledger.alerted.add(item[1])
    ledger.save()
    return "\n".join(lines)


def build_bot(pool: LlmPool, publish: bool, quiet: bool, samples: int) -> FutureEvalBot:
    return FutureEvalBot(
        pool=pool,
        quiet=quiet,
        research_reports_per_question=1,
        predictions_per_research_report=samples,
        use_research_summary_to_forecast=False,
        enable_summarize_research=False,
        publish_reports_to_metaculus=publish,
        folder_to_save_reports_to=None,
        skip_previously_forecasted_questions=True,
        extra_metadata_in_explanation=True,
        required_successful_predictions=0.4,
        llms={"default": None, "summarizer": None, "researcher": None, "parser": None},
    )


async def check_models(pool: LlmPool) -> None:
    """One tiny call per configured model plus one Tavily search; prints OK/FAILED per item."""
    for spec in pool.specs:
        # Direct call (not through the pool) so the provider's raw error message is visible.
        # Safe for public logs: the prompt is trivial and no question data is involved.
        try:
            pool.ledger.add(spec.label)
            text = await pool._make_llm(spec, spec.timeout).invoke("Reply with the single word: ready")
            print(f"  {spec.label}: OK ({text.strip()[:30]!r})")
        except Exception as error:  # noqa: BLE001
            print(f"  {spec.label}: FAILED {type(error).__name__}: {str(error)[:500]}")
    if os.getenv("TAVILY_API_KEY"):
        try:
            results = await asyncio.to_thread(tavily_search, "Metaculus forecasting tournament", max_results=2)
            print(f"  tavily: OK ({len(results)} results)")
        except Exception as error:  # noqa: BLE001
            print(f"  tavily: FAILED {type(error).__name__}: {str(error)[:200]}")
    else:
        print("  tavily: NOT CONFIGURED (TAVILY_API_KEY missing) - research will have no live search")
    pool.ledger.save()


def main() -> None:
    parser = argparse.ArgumentParser(description="FutureEval forecasting bot")
    parser.add_argument("--mode", choices=["tournament", "test_questions", "check_models"], default="tournament")
    parser.add_argument("--samples", type=int, default=int(os.getenv("BOT_SAMPLES") or "5"))
    parser.add_argument(
        "--time-budget-min",
        type=float,
        default=float(os.getenv("BOT_TIME_BUDGET_MIN") or "45"),
        help="wall-clock budget from process start (the job timeout is 55 min)",
    )
    parser.add_argument("--no-publish", action="store_true")
    parser.add_argument("--max-questions", type=int, default=6, help="test_questions mode only")
    args = parser.parse_args()
    mode: Literal["tournament", "test_questions", "check_models"] = args.mode

    quiet = mode == "tournament"
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    if quiet:
        quiet_forecast_logging()

    if not os.getenv("METACULUS_TOKEN"):
        print("METACULUS_TOKEN is not set.")
        sys.exit(1)

    pool = load_pool_from_env(DEFAULT_MODEL_SPECS)
    print(f"Models available: {[s.label for s in pool.specs]}")
    if not pool.specs:
        print("No LLM configured (set GEMINI_API_KEY and/or other provider keys).")
        sys.exit(1)

    if mode == "check_models":
        asyncio.run(check_models(pool))
        return

    publish = not args.no_publish
    if mode == "test_questions":
        bot = build_bot(pool, publish=publish, quiet=False, samples=args.samples)
        bot.skip_previously_forecasted_questions = False
        # One question per type, to test every code path without burning the daily free quota.
        questions = MetaculusClient().get_all_open_questions_from_tournament(BOT_TESTING_AREA)
        by_type: dict[str, MetaculusQuestion] = {}
        for question in questions:
            by_type.setdefault(type(question).__name__, question)
        chosen = list(by_type.values())[: args.max_questions]
        print(f"bot-testing-area: {len(questions)} open; testing types {[type(q).__name__ for q in chosen]}")
        try:
            reports = asyncio.run(bot.forecast_questions(chosen, return_exceptions=True))
        finally:
            pool.ledger.save()
        print(f"Pool stats: {pool.stats()}")
        bot.log_report_summary(reports)
        return

    # Tournament mode publishes from the runner (forecast, then comment, with recovery).
    bot = build_bot(pool, publish=False, quiet=True, samples=args.samples)
    hard_end = PROCESS_START + args.time_budget_min * 60
    try:
        summary = asyncio.run(
            run_tournaments(
                bot,
                [FALL_2026_TOURNAMENT_ID, MINIBENCH_ID],
                hard_end=hard_end,
                max_samples=args.samples,
                publish=publish,
            )
        )
    finally:
        pool.ledger.save()
    print(
        f"Done: forecast={len(summary['forecast'])} failed={len(summary['failed'])} "
        f"skipped={len(summary['skipped'])} open_total={summary['open']}"
    )
    print(f"Pool stats: {pool.stats()}")
    alert = _format_alert(summary, pool)
    if alert:
        post_alert(alert)


if __name__ == "__main__":
    main()
