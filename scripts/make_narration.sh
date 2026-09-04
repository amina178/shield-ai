#!/usr/bin/env bash
#
# Shield-AI — generate the presentation narration with macOS's built-in speech
# synthesiser. No installs, no recording, no microphone.
#
#     cd ~/Documents/AlpacaTradeHackathon
#     bash scripts/make_narration.sh
#
# Produces one audio file per slide in docs/narration/. Claude then assembles
# them with the slide images into an MP4.
#
# To hear a voice before committing to it:
#     say -v Samantha "Risk management is the killer app for agentic finance."
#     say -v Daniel   "Risk management is the killer app for agentic finance."
# List everything installed:  say -v '?' | grep en_

set -euo pipefail

VOICE="${VOICE:-Samantha}"     # override:  VOICE=Daniel bash scripts/make_narration.sh
RATE="${RATE:-170}"            # words per minute; 160-180 reads as measured, not rushed
OUT="docs/narration"

mkdir -p "$OUT"
rm -f "$OUT"/slide-*.aiff

say -v "$VOICE" -r "$RATE" -o "$OUT/slide-01.aiff" \
"Most A.I. trading agents are one bad headline away from blowing up an account. \
This is Shield A.I. It does not predict the market. It defends the downside."

say -v "$VOICE" -r "$RATE" -o "$OUT/slide-02.aiff" \
"Hand a language model a brokerage key and tell it to pick winners, and you have not \
built a quant system. You have built a leveraged guess. No covariance matrix. No tail \
estimate. No loss budget. The model is confident, not calibrated — and confidence is \
not a risk framework."

say -v "$VOICE" -r "$RATE" -o "$OUT/slide-03.aiff" \
"The design came from the scoring rule. Performance is judged on terminal account \
equity, with no benchmark. In a five day window, a thirty percent drawdown is \
unrecoverable. So maximising final equity is not a prediction problem. It is a risk \
management problem."

say -v "$VOICE" -r "$RATE" -o "$OUT/slide-04.aiff" \
"Five layers. Alpaca market data, news, and option chains feed a Python quant engine \
that computes ninety nine percent Value at Risk three independent ways, plus \
conditional VaR and beta weighted delta. The cognitive layer proposes. A deterministic \
policy filter disposes. Execution runs through Alpaca's M.C.P. server."

say -v "$VOICE" -r "$RATE" -o "$OUT/slide-05.aiff" \
"GARCH forecasts realised volatility. Options imply their own. The gap between them is \
the variance risk premium, and it is both the risk metric and the trading signal. One \
model, two jobs. When implied volatility is rich, the agent sells premium. When it is \
cheap, it buys protection. When Value at Risk breaches the mandate, it hedges \
regardless of how attractive the premium looks."

say -v "$VOICE" -r "$RATE" -o "$OUT/slide-06.aiff" \
"Why a language model at all? Because GARCH cannot read that the F.D.A. decision is on \
Thursday. Selling volatility into a known catalyst is a classic way to destroy an \
account. So the news agent holds a veto, never a vote."

say -v "$VOICE" -r "$RATE" -o "$OUT/slide-07.aiff" \
"On the first live run, the agent reported a fifteen point six volatility point edge on \
S.P.Y. A spectacular edge on the most efficiently priced instrument on earth is not an \
edge. It is a bug. It was mine. I was taking the median implied volatility across the \
whole option chain, but far out of the money puts are crash insurance and sit high on \
the smile. I was measuring skew and calling it alpha."

say -v "$VOICE" -r "$RATE" -o "$OUT/slide-08.aiff" \
"The official run used a fresh one hundred thousand dollar account. Final equity: one \
hundred thousand, one hundred and twenty six dollars. The stock book lost eighty \
dollars. The protective puts gained two hundred and seven. The hedge was the entire \
profit, and it paid because implied volatility rose toward the GARCH forecast, not \
because the underlying fell. Four other trades were refused, each with its reason \
written to the log."

say -v "$VOICE" -r "$RATE" -o "$OUT/slide-09.aiff" \
"A record of executed trades proves an agent traded. A record of refusals proves it had \
judgement. Risk management, not alpha, is the killer app for agentic finance."

echo
echo "Generated in $OUT:"
ls -lh "$OUT"/slide-*.aiff | awk '{print "  " $9 "  " $5}'
echo
echo "Total spoken length:"
afinfo "$OUT"/slide-*.aiff 2>/dev/null | awk '/estimated duration/ {s+=$3} END {printf "  %.0f seconds (%.1f minutes)\n", s, s/60}'
echo
echo "Listen to one:   afplay $OUT/slide-01.aiff"
echo "Different voice: VOICE=Daniel bash scripts/make_narration.sh"
