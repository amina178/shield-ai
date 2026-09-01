"""
Shield-AI — News Sentiment Agent.

This is the answer to the question every judge will ask: why does a quantitative
strategy need a language model at all?

Because GARCH cannot read that the FDA decision is on Thursday.

A volatility model sees only a series of numbers. It has no idea that the option
it is about to sell expires the day after an earnings release. Selling volatility
into a known catalyst is one of the classic ways to destroy an account: implied
volatility is high precisely BECAUSE the market knows something is coming, so the
"edge" the model sees is not an edge at all — it is compensation for a risk the
model cannot perceive. The premium looks free right up until the gap.

So this agent holds a VETO, not a vote. It never proposes a trade and never
sizes one. It answers exactly one question: is there event risk on this
underlying that the volatility model cannot see? If yes, the trade is blocked
regardless of how attractive the numbers look.

The agent degrades gracefully. With an LLM key it reads and reasons over
headlines. Without one it falls back to a deterministic keyword scanner. The
fallback is deliberately BLUNT and errs towards vetoing: for a risk system, a
missed opportunity costs a few dollars of premium and a missed catalyst costs a
drawdown you cannot earn back inside a five-day scoring window.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

# Event categories that make selling short-dated premium unwise. These are
# scheduled or announced events whose resolution produces a gap rather than a
# drift — exactly the move a delta-based risk model underestimates.
EVENT_PATTERNS: dict[str, list[str]] = {
    "earnings": [
        r"\bearnings\b", r"\bq[1-4]\s+results\b", r"\bquarterly results\b",
        r"\breports? (?:earnings|results)\b", r"\bbeats?\b.*\bestimates?\b",
        r"\bmisses?\b.*\bestimates?\b", r"\bguidance\b", r"\bpre-?announce",
    ],
    "regulatory": [
        r"\bfda\b", r"\bapprov(?:al|es|ed)\b", r"\bclinical trial\b",
        r"\bphase [123]\b", r"\bsec (?:probe|investigation|charges)\b",
        r"\bantitrust\b", r"\bdoj\b", r"\bregulator", r"\bsubpoena\b",
    ],
    "corporate_action": [
        r"\bmerger\b", r"\bacquisition\b", r"\bacquires?\b", r"\bbuyout\b",
        r"\btakeover\b", r"\bspin-?off\b", r"\bstock split\b",
        r"\bbankrupt", r"\bchapter 11\b", r"\bdelist",
    ],
    "leadership": [
        r"\bceo\b.*\b(?:resign|step down|depart|ousted|fired)\b",
        r"\bcfo\b.*\b(?:resign|step down|depart|ousted|fired)\b",
        r"\bnames? new (?:ceo|cfo)\b",
    ],
    "shock": [
        r"\brecall\b", r"\bdata breach\b", r"\bcyber ?attack\b",
        r"\bplunge", r"\bhalt(?:ed|s)? trading\b", r"\bfraud\b",
        r"\bshort seller report\b", r"\bprofit warning\b",
    ],
}

_COMPILED = {
    cat: [re.compile(p, re.IGNORECASE) for p in pats]
    for cat, pats in EVENT_PATTERNS.items()
}


@dataclass(frozen=True)
class SentimentVerdict:
    """The agent's answer for one underlying."""

    symbol: str
    has_event_risk: bool
    categories: list[str] = field(default_factory=list)
    veto_reason: str = ""
    severity: int = 0                  # 0 none, 1 watch, 2 elevated, 3 blocking
    headlines_reviewed: int = 0
    evidence: list[str] = field(default_factory=list)
    source: str = "keyword"            # "llm" or "keyword"
    rationale: str = ""

    @property
    def blocks_premium_selling(self) -> bool:
        """Selling premium into an unresolved catalyst is the forbidden trade."""
        return self.severity >= 2


# ---------------------------------------------------------------------------
# Deterministic fallback
# ---------------------------------------------------------------------------

def scan_headlines(symbol: str, news: list[dict]) -> SentimentVerdict:
    """
    Keyword scan over recent headlines for this underlying.

    Runs with no API key and no network beyond the Alpaca news call that already
    happened. It is crude — it cannot tell "Apple beat earnings last week"
    (resolved, harmless) from "Apple reports earnings Thursday" (unresolved,
    dangerous) — so it treats both as risk. That asymmetry is intentional.
    """
    relevant = [
        n for n in news
        if symbol in (n.get("symbols") or [])
    ]

    hits: dict[str, list[str]] = {}
    for item in relevant:
        text = f"{item.get('headline', '')} {item.get('summary', '')}"
        for category, patterns in _COMPILED.items():
            if any(p.search(text) for p in patterns):
                hits.setdefault(category, []).append(
                    item.get("headline", "")[:120]
                )

    if not hits:
        return SentimentVerdict(
            symbol=symbol, has_event_risk=False,
            headlines_reviewed=len(relevant), source="keyword",
            rationale="no event-risk keywords in recent headlines",
        )

    # Earnings and regulatory events are the ones that gap. Weight them highest.
    severity = 3 if ({"earnings", "regulatory"} & hits.keys()) else 2
    categories = sorted(hits)
    evidence = [h for hs in hits.values() for h in hs][:3]

    return SentimentVerdict(
        symbol=symbol,
        has_event_risk=True,
        categories=categories,
        severity=severity,
        veto_reason=(
            f"event risk ({', '.join(categories)}) detected in "
            f"{len(relevant)} recent headline(s)"
        ),
        headlines_reviewed=len(relevant),
        evidence=evidence,
        source="keyword",
        rationale=f"keyword match on {', '.join(categories)}",
    )


# ---------------------------------------------------------------------------
# LLM path
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """You are the risk-screening layer of an autonomous options \
trading agent. The agent sells short-dated option premium when implied \
volatility is rich relative to a GARCH forecast of realised volatility.

Your ONLY job is to decide whether an identifiable, UNRESOLVED catalyst sits \
inside the life of the option. You never propose trades and never size them.

What matters is timing, not tone. A stock that already fell 8% on news that has \
now been digested is not event risk. A stock reporting earnings in three days \
is, however positive the coverage.

Score severity:
  0 = no identifiable catalyst
  1 = background noise, analyst chatter, routine coverage
  2 = a catalyst is plausible but its timing is unclear
  3 = a scheduled or announced catalyst is likely unresolved

Reply with JSON only:
{"severity": 0-3, "categories": ["earnings"|"regulatory"|"corporate_action"|\
"leadership"|"shock"], "reason": "one sentence", "evidence": ["headline", ...]}"""


def analyse_with_llm(
    symbol: str,
    news: list[dict],
    horizon_days: int,
    api_key: str | None = None,
    model: str = "claude-sonnet-4-20250514",
) -> SentimentVerdict | None:
    """
    Ask the language model to judge event risk. Returns None if unavailable, so
    the caller falls back to the keyword scan rather than failing the cycle.

    A trading agent that stops trading because an LLM endpoint is slow has
    confused its cognitive layer with its critical path.
    """
    key = api_key or os.getenv("ANTHROPIC_API_KEY", "")
    if not key:
        return None

    relevant = [n for n in news if symbol in (n.get("symbols") or [])]
    if not relevant:
        return SentimentVerdict(
            symbol=symbol, has_event_risk=False, headlines_reviewed=0,
            source="llm", rationale="no headlines for this symbol",
        )

    try:
        import anthropic
    except ImportError:
        return None

    digest = "\n".join(
        f"- [{n.get('created_at', '')[:10]}] {n.get('headline', '')}"
        for n in relevant[:25]
    )
    user = (
        f"Underlying: {symbol}\n"
        f"Option expires in {horizon_days} days.\n\n"
        f"Recent headlines:\n{digest}"
    )

    try:
        client = anthropic.Anthropic(api_key=key)
        resp = client.messages.create(
            model=model,
            max_tokens=400,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user}],
        )
        text = "".join(
            block.text for block in resp.content if block.type == "text"
        ).strip()

        # Models sometimes wrap JSON in prose or a fence; extract defensively.
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None
        parsed = json.loads(match.group(0))
    except Exception:  # noqa: BLE001
        # Any failure — network, rate limit, malformed JSON — falls back.
        return None

    severity = int(parsed.get("severity", 0))
    categories = list(parsed.get("categories", []))
    reason = str(parsed.get("reason", ""))

    return SentimentVerdict(
        symbol=symbol,
        has_event_risk=severity >= 2,
        categories=categories,
        severity=severity,
        veto_reason=reason if severity >= 2 else "",
        headlines_reviewed=len(relevant),
        evidence=[str(e)[:120] for e in parsed.get("evidence", [])][:3],
        source="llm",
        rationale=reason,
    )


def assess(
    symbol: str,
    news: list[dict],
    horizon_days: int = 14,
    use_llm: bool = True,
) -> SentimentVerdict:
    """
    Public entry point: LLM when available, keyword scan otherwise.

    The two paths are combined conservatively — if the keyword scanner sees a
    catalyst the model missed, the higher severity wins. The cognitive layer may
    add judgement, but it may not remove a warning the deterministic layer
    raised.
    """
    keyword = scan_headlines(symbol, news)

    if use_llm:
        llm = analyse_with_llm(symbol, news, horizon_days)
        if llm is not None:
            if keyword.severity > llm.severity:
                return SentimentVerdict(
                    symbol=symbol,
                    has_event_risk=keyword.has_event_risk,
                    categories=sorted(set(keyword.categories) | set(llm.categories)),
                    severity=keyword.severity,
                    veto_reason=keyword.veto_reason,
                    headlines_reviewed=keyword.headlines_reviewed,
                    evidence=keyword.evidence,
                    source="llm+keyword",
                    rationale=(
                        f"LLM said severity {llm.severity} "
                        f"({llm.rationale}); keyword scan escalated to "
                        f"{keyword.severity}"
                    ),
                )
            return llm

    return keyword
