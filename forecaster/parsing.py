"""
Deterministic parsers for LLM forecast outputs.

The free-tier LLM budget is tight, so we parse the model's final answer with regexes instead
of spending a Flash call per sample on an LLM parser. When a parse fails, main.py asks a cheap
Flash-Lite model to rewrite just the final answer in the exact format and parses that.

Design rules (from review):
- bind to the FINAL answer: never silently fall back to a draft/base-rate number in the reasoning;
- never silently truncate a number (e.g. '3.75B' -> 3): reject instead;
- multiple-choice labels must match an option exactly (after normalisation), all options from
  one contiguous block of lines.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone


class ParseError(ValueError):
    pass


# ----------------------------------------------------------------------------- normalisation

_TRANSLATE = str.maketrans(
    {
        "−": "-",  # minus sign
        "–": "-",  # en dash
        "—": "-",  # em dash
        "‘": "'",
        "’": "'",
        "“": '"',
        "”": '"',
        " ": " ",  # nbsp
        " ": " ",  # thin space
        " ": " ",
        "*": "",  # markdown bold/italic/bullets
    }
)


def normalize(text: str) -> str:
    return text.translate(_TRANSLATE)


def _label_key(label: str) -> str:
    label = normalize(label).casefold()
    label = re.sub(r"\s+", " ", label).strip()
    label = label.strip("\"'`")
    label = label.rstrip(".").strip()
    return label


_NUM = r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:[eE][-+]?\d+)?|[-+]?\.\d+"


def _to_float(text: str) -> float:
    value = float(text.replace(",", ""))
    if not math.isfinite(value):
        raise ParseError("non-finite number")
    return value


# ----------------------------------------------------------------------------- binary

_PROB_LINE = re.compile(
    r"probability\b[^\n:=]{0,30}?[:=]\s*(?:~|approx\.?|approximately|about|around)?\s*("
    + _NUM
    + r")\s*(%|percent)?",
    re.IGNORECASE,
)
_PROB_STRICT = re.compile(r"probability\s*[:=]\s*(" + _NUM + r")\s*%", re.IGNORECASE)


def _probability_from_match(number: str, unit: str | None) -> float:
    value = _to_float(number)
    if not unit and value <= 1.0:
        probability = value
    else:
        probability = value / 100.0
    if not 0.0 <= probability <= 1.0:
        raise ParseError("probability out of range")
    return probability


def parse_binary_probability(text: str) -> float:
    """Parse the final 'Probability: ZZ%' line. Only the last few lines count."""
    text = normalize(text)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines[-4:]):
        if "probability" not in line.lower():
            continue
        if re.search(r"\d\s*%?\s*-\s*\d", line.split(":", 1)[-1]):
            raise ParseError("final probability line contains a range")
        match = _PROB_LINE.search(line)
        if not match:
            raise ParseError("final probability line has no readable number")
        return _probability_from_match(match.group(1), match.group(2))
    matches = list(_PROB_STRICT.finditer(text))
    if not matches:
        raise ParseError("no final 'Probability: XX%' line")
    return _probability_from_match(matches[-1].group(1), "%")


# ----------------------------------------------------------------------------- multiple choice

_MC_LINE = re.compile(
    r"^(.*?)\s*[:=]\s*(" + _NUM + r")\s*(%|percent)?\s*(?:\([^()]*\))?\s*\.?$", re.IGNORECASE
)


def _strip_decoration(line: str) -> str:
    line = line.strip()
    line = re.sub(r"^(?:[-•>]\s+)", "", line)
    line = re.sub(r"^\d{1,2}[.)]\s+", "", line)
    return line.strip().strip("\"'`").strip()


def _mc_line(line: str, option_keys: dict[str, str]) -> tuple[str, float, bool] | None:
    stripped = line.strip()
    if stripped.startswith("|"):
        cells = [c.strip() for c in stripped.strip("|").split("|")]
        cells = [c for c in cells if c]
        if len(cells) < 2:
            return None
        label, value_text = cells[0], cells[-1]
        value_match = re.fullmatch(r"(" + _NUM + r")\s*(%|percent)?", value_text, re.IGNORECASE)
        if not value_match:
            return None
        candidates = [label]
        number, pct = value_match.group(1), value_match.group(2)
    else:
        body = _strip_decoration(stripped)
        match = _MC_LINE.match(body)
        if not match:
            return None
        candidates = [match.group(1)]
        number, pct = match.group(2), match.group(3)
    for label in list(candidates):
        candidates.append(re.sub(r"^option\s*[:\-]?\s*", "", label, flags=re.IGNORECASE))
    for label in candidates:
        key = _label_key(label)
        if key in option_keys:
            return option_keys[key], _to_float(number), bool(pct)
    return None


def parse_multiple_choice(text: str, options: list[str]) -> dict[str, float]:
    """
    Finds the LAST block of consecutive lines that assigns a value to every option
    (labels must equal an option after normalisation) and returns normalised probabilities.
    """
    option_keys = {_label_key(o): o for o in options}
    if len(option_keys) != len(options):
        raise ParseError("options are not distinguishable after normalisation")
    # blank lines are kept so that paragraphs of reasoning separate blocks
    lines = normalize(text).splitlines()
    parsed = [(_mc_line(line, option_keys) if line.strip() else None, i) for i, line in enumerate(lines)]
    hits = [(i, hit) for hit, i in parsed if hit is not None]
    if not hits:
        raise ParseError("no option probabilities found")
    # group hits into blocks of nearby lines (at most 1 non-matching line, e.g. a blank, between)
    blocks: list[list[tuple[int, tuple[str, float, bool]]]] = []
    for index, hit in hits:
        if blocks and index - blocks[-1][-1][0] <= 2:
            blocks[-1].append((index, hit))
        else:
            blocks.append([(index, hit)])
    for block in reversed(blocks):
        values: dict[str, tuple[float, bool]] = {}
        for _, (option, value, pct) in block:
            values[option] = (value, pct)
        if set(values) == set(options):
            any_pct = any(pct for _, pct in values.values())
            all_small = all(value <= 1.0 for value, _ in values.values())
            scale = 1.0 if (not any_pct and all_small) else 100.0
            probs = {o: values[o][0] / scale for o in options}
            if any(p < 0 for p in probs.values()):
                raise ParseError("negative option probability")
            total = sum(probs.values())
            if not 0.9 <= total <= 1.1:
                raise ParseError("option probabilities do not sum to ~100%")
            return {o: p / total for o, p in probs.items()}
    raise ParseError("no block lists every option")


# ----------------------------------------------------------------------------- numeric

_SCALES = {"thousand": 1e3, "k": 1e3, "million": 1e6, "mn": 1e6, "m": 1e6, "billion": 1e9, "bn": 1e9, "b": 1e9, "trillion": 1e12, "t": 1e12}

_VALUE = (
    r"(?:[$€£¥]|usd|eur|gbp)?\s*("
    + _NUM
    + r")(?:\s*(thousand|million|billion|trillion|bn|mn|[kKMBT])(?![A-Za-z]))?"
    + r"(?!\w|\.\d|,\d|\s*-\s*\d|\s\d{3}\b|'\d)"
)
_LEVEL = r"(\d{1,2}(?:\.\d+)?)(?:st|nd|rd|th)?"
_PERCENTILE_PATTERNS = [
    re.compile(r"percentile\s*" + _LEVEL + r"\s*[:=|]\s*" + _VALUE, re.IGNORECASE),
    re.compile(_LEVEL + r"\s*(?:th\s*)?percentile\s*[:=|]\s*" + _VALUE, re.IGNORECASE),
    re.compile(r"\bp" + _LEVEL + r"\s*[:=|]\s*" + _VALUE, re.IGNORECASE),
    re.compile(r"^\|?\s*(?:p|percentile\s*)?" + _LEVEL + r"(?:\s*percentile)?\s*\|\s*" + _VALUE, re.IGNORECASE),
]


def unit_scale(unit: str | None) -> float | None:
    if not unit:
        return None
    lowered = unit.lower()
    for word, scale in (("trillion", 1e12), ("billion", 1e9), ("million", 1e6), ("thousand", 1e3)):
        if word in lowered:
            return scale
    tokens = re.findall(r"[A-Za-z]+", unit)
    for token in tokens:
        if token in ("bn", "B"):
            return 1e9
        if token in ("mn", "M", "MM"):
            return 1e6
        if token in ("k", "K"):
            return 1e3
    return None


def _apply_scale(value: float, suffix: str | None, unit: str | None) -> float:
    if not suffix:
        return value
    key = suffix if suffix in ("K", "M", "B", "T") else suffix.lower()
    suffix_scale = _SCALES.get(key.lower())
    if suffix_scale is None:
        return value
    target = unit_scale(unit)
    if target is None:
        return value * suffix_scale
    return value * suffix_scale / target


def parse_percentiles(text: str, expected: list[float], unit: str | None = None) -> list[tuple[float, float]]:
    """
    Parses percentile lines ('Percentile 10: 123', '10th percentile: 123', 'P10: 123', table rows).
    The last occurrence of each level wins. Returns sorted (percentile_in_0_1, value) pairs.
    """
    text = normalize(text)
    found: dict[float, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        for pattern in _PERCENTILE_PATTERNS:
            match = pattern.search(line)
            if match:
                level = float(match.group(1))
                value = _apply_scale(_to_float(match.group(2)), match.group(3), unit)
                found[level] = value
                break
    missing = [p for p in expected if float(p) not in found]
    if missing:
        raise ParseError(f"missing {len(missing)} percentile level(s)")
    ordered = [(p / 100.0, found[float(p)]) for p in sorted(expected)]
    values = [v for _, v in ordered]
    if any(values[i] > values[i + 1] for i in range(len(values) - 1)):
        raise ParseError("percentile values decrease")
    return ordered


# ----------------------------------------------------------------------------- dates

_DATE_VALUE = (
    r"(\d{4}-\d{1,2}-\d{1,2}(?:[T ]\d{1,2}:\d{2}(?::\d{2})?Z?)?"
    r"|[A-Za-z]{3,9}\.? \d{1,2},? \d{4}"
    r"|\d{1,2} [A-Za-z]{3,9},? \d{4})"
)
_DATE_PATTERNS = [
    re.compile(r"percentile\s*" + _LEVEL + r"\s*[:=|]\s*" + _DATE_VALUE, re.IGNORECASE),
    re.compile(_LEVEL + r"\s*(?:th\s*)?percentile\s*[:=|]\s*" + _DATE_VALUE, re.IGNORECASE),
    re.compile(r"\bp" + _LEVEL + r"\s*[:=|]\s*" + _DATE_VALUE, re.IGNORECASE),
]


def _parse_date(value: str) -> float:
    value = value.strip().replace(",", "")
    iso = re.fullmatch(r"(\d{4})-(\d{1,2})-(\d{1,2})(?:[T ](\d{1,2}):(\d{2})(?::(\d{2}))?Z?)?", value)
    if iso:
        y, mo, d, h, mi, s = iso.groups()
        dt = datetime(int(y), int(mo), int(d), int(h or 0), int(mi or 0), int(s or 0), tzinfo=timezone.utc)
        return dt.timestamp()
    for fmt in ("%B %d %Y", "%b %d %Y", "%b. %d %Y", "%d %B %Y", "%d %b %Y"):
        try:
            return datetime.strptime(value, fmt).replace(tzinfo=timezone.utc).timestamp()
        except ValueError:
            continue
    raise ParseError("unreadable date")


def parse_date_percentiles(text: str, expected: list[float]) -> list[tuple[float, float]]:
    """Parses date percentiles. Returns (percentile_in_0_1, unix_timestamp)."""
    text = normalize(text)
    found: dict[float, float] = {}
    for line in text.splitlines():
        for pattern in _DATE_PATTERNS:
            match = pattern.search(line)
            if match:
                found[float(match.group(1))] = _parse_date(match.group(2))
                break
    missing = [p for p in expected if float(p) not in found]
    if missing:
        raise ParseError(f"missing {len(missing)} date percentile level(s)")
    ordered = [(p / 100.0, found[float(p)]) for p in sorted(expected)]
    values = [v for _, v in ordered]
    if any(values[i] > values[i + 1] for i in range(len(values) - 1)):
        raise ParseError("date percentiles are not chronological")
    return ordered
