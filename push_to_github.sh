#!/usr/bin/env bash
#
# Shield-AI — one-shot GitHub setup.
#
# Run this ONCE from inside the project folder:
#
#     cd ~/Documents/AlpacaTradeHackathon
#     bash push_to_github.sh
#
# It initialises git, makes the first commit, creates the public repository
# and pushes. Safe to re-run: it detects what already exists and skips it.

set -euo pipefail

REPO_NAME="shield-ai"
REPO_DESC="Autonomous Risk Guardrails & Hedging Agent — Lablab.ai x Alpaca Hackathon"

# --- sanity check: are we in the right folder? -----------------------------
if [ ! -d "shield_ai" ]; then
  echo "ERROR: run this from the project folder (the one containing shield_ai/)."
  exit 1
fi

# --- guard: never commit real credentials ----------------------------------
if [ -f ".env" ] && ! grep -q "^\.env$" .gitignore 2>/dev/null; then
  echo "ERROR: .env exists but is not git-ignored. Aborting to protect your keys."
  exit 1
fi

# --- git identity ----------------------------------------------------------
if ! git config --get user.email >/dev/null 2>&1; then
  git config --global user.email "amina.baibeck@gmail.com"
  git config --global user.name "Amina Baibek"
  echo "Set git identity."
fi

# --- initialise ------------------------------------------------------------
if [ ! -d ".git" ]; then
  git init -b main
  echo "Initialised empty repository on branch main."
else
  echo "Repository already initialised."
  git branch -M main 2>/dev/null || true
fi

git add -A
if git diff --cached --quiet; then
  echo "Nothing new to commit."
else
  git commit -m "Shield-AI: quantitative risk core

Deterministic risk engine that defines the boundary the LLM layer operates
inside:

- Portfolio VaR via three independent estimators (parametric with Student-t
  tails, historical simulation, Monte Carlo) plus Conditional VaR
- Beta-weighted net Delta so exposures across names are actually additive
- GARCH(1,1) volatility forecaster, reused as the variance-risk-premium
  signal against option-implied volatility
- Kupiec proportion-of-failures and Christoffersen independence tests, so the
  risk model is statistically validated rather than asserted
- 12 verification tests against synthetic data with analytically known answers"
  echo "Created first commit."
fi

# --- create the remote -----------------------------------------------------
if git remote get-url origin >/dev/null 2>&1; then
  echo "Remote 'origin' already set: $(git remote get-url origin)"
else
  if command -v gh >/dev/null 2>&1; then
    echo "Creating public repository via gh..."
    gh repo create "$REPO_NAME" --public --source=. --remote=origin \
      --description "$REPO_DESC"
  else
    cat <<'EOF'

The GitHub CLI (gh) is not installed, so create the repository manually:

  1. Open https://github.com/new
  2. Name it:  shield-ai
  3. Visibility: Public
  4. Do NOT tick "Add a README" or "Add a license" — you already have both
  5. Click "Create repository", then run:

     git remote add origin https://github.com/amina178/shield-ai.git
     git push -u origin main

Alternatively install gh first:  brew install gh && gh auth login
Then re-run this script.

EOF
    exit 0
  fi
fi

# --- push ------------------------------------------------------------------
git push -u origin main
echo
echo "Done: https://github.com/amina178/$REPO_NAME"
