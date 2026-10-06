# metaculus-forecast-bot

An open-source, hobbyist forecasting bot for the [Metaculus Fall 2026 FutureEval](https://www.metaculus.com/tournament/fall-futureeval-2026/) bot tournament and the bi-weekly [MiniBench](https://www.metaculus.com/tournament/minibench/).

It is built on Metaculus' [metac-bot-template](https://github.com/Metaculus/metac-bot-template) and the [`forecasting-tools`](https://github.com/Metaculus/forecasting-tools) package. The code was written with the help of an AI coding agent (Claude Code). The bot runs fully automatically, and no human is in the loop on individual forecasts.

## How it works

For each open question the bot has not forecast yet, it does the following, starting with the questions that close soonest:

1. **Research.** Gemini 3.x has no free search grounding, so research works in three steps:
   1. A cheap Flash-Lite model writes 2 search queries: one for recent news and one for base rates and history.
   2. [Tavily](https://tavily.com) runs both searches (free tier).
   3. The Flash-Lite model writes a short brief from the results. The brief covers:
      - the current status, with dates and sources
      - what the resolution source shows
      - **base rates**
      - upcoming events
      - prediction-market, poll and expert signals
      - gaps in the evidence

   The raw search results are kept too, so the forecasters can check claims. Optionally it adds AskNews headlines.
2. **Forecast (up to N independent samples, default 5).** Each sample is a separate call. The calls are spread across the free Gemini Flash models (3.8, 3.7, 3.6, 3.5, 3-preview), which makes a cheap ensemble. Each prompt asks for status quo, base rate, an evidence update, market signals, and the best case for each side. Answers are parsed deterministically with regexes, so no extra LLM "parser" calls are needed. When the day's remaining free quota runs low, the bot takes fewer strong samples rather than padding with weaker models.
3. **Aggregate.** How samples are combined depends on the question type:
   - **Binary:** median of the samples, clipped to [1.5%, 98.5%].
   - **Multiple choice:** mean of the samples, with a 0.5% floor per option, then renormalized.
   - **Numeric, discrete and date:** the samples' CDFs are averaged point by point (a mixture distribution). The bot elicits 13 percentiles from 2% to 98%, so the tails are represented.
4. **Publish.** The forecast is posted together with a **private** comment that contains the research and every sample's reasoning, as the tournament rules require.

## Operations

- **Triggering.** GitHub's built-in cron has been unreliable, so an external scheduler (cron-job.org) calls `workflow_dispatch` on `run_bot_on_tournament.yaml` every 10 minutes. GitHub's own cron stays on as a backup.
- **Idempotent runs.** The bot only forecasts questions it hasn't forecast yet, according to Metaculus. In the tournament it never re-forecasts or reruns a question to change a forecast.
- **Bounded retries.** A question that fails is retried at most once per day, with fewer samples, after all never-attempted questions. A malformed final answer is first rescued by a cheap Flash-Lite call that restates it in the exact format. Conditional questions forecast only the yes and no branches, using the library default.
- **Free-tier limits.** Each model call is paced to stay under its per-minute limit. A per-model daily usage ledger is carried between runs in the Actions cache and resets at midnight Pacific, when Google resets quotas. A model that hits its daily quota is skipped until the reset. The main tournament is served before MiniBench.
- **No forecast values in public logs.** This repo is public, so tournament-mode logs show only counts and question links. Forecasts and reasoning appear only in the private Metaculus comments.
- **Alerts.** Failures or skipped questions open or update a `bot-alert` GitHub issue. The issue holds only links, counts and sanitized error types.
- **Testing.** Tests run only on the public [bot-testing-area](https://www.metaculus.com/tournament/bot-testing-area/), via the `Test Bot` workflow, and with offline unit tests in `tests/`. The bot maker does not look at forecasts on open tournament questions to tune the bot.

## Configuration

| Name | Kind | Purpose |
|---|---|---|
| `METACULUS_TOKEN` | secret | Bot account token (required) |
| `GEMINI_API_KEY` | secret | Google AI Studio key (free tier) |
| `TAVILY_API_KEY` | secret | Tavily search key (free tier, 1,000 searches/month) |
| `GROQ_API_KEY` | secret | Optional free fallback model |
| `OPENROUTER_API_KEY` | secret | Optional; e.g. Metaculus-provided credits (enable models via `BOT_MODEL_POOL`) |
| `ASKNEWS_CLIENT_ID` / `ASKNEWS_SECRET` | secret | Optional news research |
| `BOT_MODEL_POOL` | variable | Optional JSON list that overrides the model pool without a code change |
| `BOT_SAMPLES` | variable | Optional samples per question (default 5) |

## Running locally

```bash
python -m venv .venv && .venv/Scripts/activate   # or source .venv/bin/activate
pip install -r requirements.txt pytest
python -m pytest -q tests                        # offline tests
python main.py --mode check_models               # one tiny call per configured model
python main.py --mode test_questions --samples 2 # forecasts on bot-testing-area
```
