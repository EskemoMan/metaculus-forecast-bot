"""
Research step: web search (Tavily free tier) + a cheap LLM summary.

Gemini 3.x has no free Google Search grounding, so searching is done with Tavily
(1,000 free credits/month; a basic search costs 1). Per question:
1. a cheap "research" model writes 2 search queries (falls back to templated queries);
2. 2 Tavily searches: recent news + general background/base rates;
3. the cheap model writes a structured brief from the results (falls back to the raw results).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
from datetime import datetime, timezone

import requests
from forecasting_tools import MetaculusQuestion, clean_indents

from forecaster.llm_pool import AllModelsFailed, LlmPool

logger = logging.getLogger(__name__)

TAVILY_URL = "https://api.tavily.com/search"


def tavily_search(query: str, *, topic: str = "general", days: int | None = None, max_results: int = 6) -> list[dict]:
    key = os.getenv("TAVILY_API_KEY")
    if not key:
        return []
    payload: dict = {
        "query": query[:380],
        "search_depth": "basic",
        "topic": topic,
        "max_results": max_results,
        "include_answer": False,
    }
    if days and topic == "news":
        payload["days"] = days
    response = requests.post(
        TAVILY_URL,
        json=payload,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        timeout=40,
    )
    response.raise_for_status()
    return response.json().get("results", [])


def _format_results(results: list[dict], limit_chars: int = 900) -> str:
    lines = []
    for item in results:
        title = (item.get("title") or "").strip()
        url = item.get("url") or ""
        date = item.get("published_date") or ""
        content = re.sub(r"\s+", " ", item.get("content") or "").strip()[:limit_chars]
        lines.append(f"- {title} ({date}) {url}\n  {content}")
    return "\n".join(lines)


def demote_headings(text: str) -> str:
    """Turn any markdown heading in LLM text into a level-4 heading so it nests under ours."""
    return re.sub(r"(?m)^[ \t]*#{1,8}[ \t]+", "#### ", text)


def _fallback_queries(question: MetaculusQuestion) -> list[str]:
    title = re.sub(r"\s+", " ", question.question_text).strip()
    return [title, f"{title} history base rate"]


async def _make_queries(pool: LlmPool, question: MetaculusQuestion) -> list[str]:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    prompt = clean_indents(
        f"""
        Today is {today}. Write 2 web search queries to research this forecasting question.
        Query 1: the latest news and current status relevant to how it will resolve.
        Query 2: historical data, base rates or official statistics/schedules for comparable events.
        Each query is under 12 words, no quotes or operators. Output exactly two lines:
        Q1: ...
        Q2: ...

        Question: {question.question_text}
        Resolution criteria: {question.resolution_criteria}
        """
    )
    try:
        text, _ = await pool.call(prompt, role="research")
    except AllModelsFailed:
        return _fallback_queries(question)
    found = re.findall(r"Q[12]\s*[:.\-]\s*(.+)", text)
    queries = [q.strip().strip('"') for q in found if q.strip()][:2]
    return queries if len(queries) == 2 else _fallback_queries(question)


async def research_question(pool: LlmPool, question: MetaculusQuestion) -> str:
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    queries = await _make_queries(pool, question)

    raw_sections: list[str] = []
    searches = [
        (queries[0], "news", 30),
        (queries[1], "general", None),
    ]
    for query, topic, days in searches:
        try:
            results = await asyncio.to_thread(tavily_search, query, topic=topic, days=days)
        except Exception as error:  # noqa: BLE001
            logger.warning(f"Tavily search failed: {type(error).__name__}")
            results = []
        if results:
            raw_sections.append(f"#### Search: {query} ({topic})\n{_format_results(results)}")

    if not raw_sections:
        return (
            "Research gathered for this question.\n#### Web research unavailable\n"
            "No search results could be retrieved. Rely on general knowledge, keep in mind it "
            "may be out of date, and stay close to base rates."
        )

    raw = "\n\n".join(raw_sections)
    prompt = clean_indents(
        f"""
        You are the research assistant for a superforecaster. Today is {today}.
        Using ONLY the search results below plus well-established background knowledge, write a
        factual research brief (at most ~500 words) for this question. Do not give a forecast.

        Question: {question.question_text}
        Resolution criteria: {question.resolution_criteria}
        Fine print: {question.fine_print}

        Sections:
        1. Current status: latest relevant facts, each with its date and source.
        2. Resolution source: what the source named in the resolution criteria shows, if present.
        3. Base rates: how often comparable events happened (numbers), if the results allow.
        4. Upcoming: scheduled events or deadlines before resolution.
        5. Market/expert signals: prediction-market odds, polls, official projections, with dates.
        6. Gaps: important things the results do not answer, and anything stale or contradictory.

        SEARCH RESULTS:
        {raw}
        """
    )
    try:
        brief, model = await pool.call(prompt, role="research")
        return (
            f"Research gathered for this question.\n#### Research brief ({model}, from web search)\n"
            f"{demote_headings(brief)}\n\n#### Raw search results\n{raw}"
        )
    except AllModelsFailed:
        return f"Research gathered for this question.\n#### Raw web search results (summary unavailable)\n{raw}"
