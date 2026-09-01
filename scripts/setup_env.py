"""
Shield-AI — interactive credential setup.

    PYTHONPATH=. python3 scripts/setup_env.py

Prompts for your Alpaca paper keys, writes .env, and immediately verifies the
credentials against the live account so you find out here — not three scripts
later — whether they work.

Why a script instead of `read` in the shell: pasting a multi-line block into a
terminal feeds the remaining pasted lines into `read` as its answer, silently
capturing a shell command as your API key. A prompt inside a running program
reads from the terminal after the paste is finished, so it cannot happen.
"""

from __future__ import annotations

import os
import stat
import sys
from getpass import getpass
from pathlib import Path

ENV_PATH = Path(".env")


def mask(value: str) -> str:
    if len(value) <= 8:
        return "*" * len(value)
    return f"{value[:4]}...{value[-4:]}"


def prompt_keys() -> tuple[str, str, str]:
    print("Alpaca paper credentials")
    print("Get them at: Alpaca dashboard -> Paper Trading -> Home -> API Keys")
    print("(the secret is shown only once when generated)\n")

    api_key = input("  API key ID          : ").strip()
    # getpass hides the secret from the screen and from any terminal scrollback
    # someone might later screenshot.
    secret = getpass("  Secret key (hidden) : ").strip()

    if not api_key or not secret:
        print("\nERROR: both values are required.")
        sys.exit(1)

    # A pasted shell command is the classic failure mode this script exists to
    # prevent — catch it explicitly rather than writing nonsense to disk.
    for name, val in (("API key", api_key), ("secret", secret)):
        if " " in val or val.startswith("read ") or "=" in val:
            print(f"\nERROR: the {name} looks wrong: {val[:40]!r}")
            print("It should be a single token with no spaces or '='.")
            sys.exit(1)

    print()
    anthropic = getpass("  Anthropic API key (optional, Enter to skip): ").strip()
    return api_key, secret, anthropic


def write_env(api_key: str, secret: str, anthropic: str) -> None:
    lines = [
        "# Shield-AI credentials. Git-ignored — never commit this file.",
        "",
        f"ALPACA_API_KEY={api_key}",
        f"ALPACA_SECRET_KEY={secret}",
        "ALPACA_PAPER_TRADE=true",
    ]
    if anthropic:
        lines += ["", f"ANTHROPIC_API_KEY={anthropic}"]

    ENV_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    # Owner read/write only. A credentials file readable by other accounts on
    # the machine is a credentials file you have already lost.
    os.chmod(ENV_PATH, stat.S_IRUSR | stat.S_IWUSR)
    print(f"\nWrote {ENV_PATH.resolve()} (permissions 600)")


def verify(api_key: str, secret: str) -> int:
    print("\nVerifying against Alpaca...")
    try:
        from alpaca.trading.client import TradingClient
    except ImportError:
        print("  alpaca-py is not installed — skipping verification.")
        print("  Run: pip install alpaca-py")
        return 0

    try:
        acct = TradingClient(api_key, secret, paper=True).get_account()
    except Exception as exc:  # noqa: BLE001
        text = f"{type(exc).__name__}: {exc}"
        print(f"  FAILED: {text}")

        # Distinguish "cannot reach Alpaca" from "Alpaca says no". Blaming the
        # keys for a network fault sends you hunting the wrong problem, which
        # is exactly the kind of wasted hour a hackathon cannot afford.
        network_markers = (
            "ProxyError", "ConnectionError", "Timeout", "Max retries",
            "NameResolution", "SSLError", "Tunnel connection failed",
        )
        if any(m.lower() in text.lower() for m in network_markers):
            print("\n  This is a NETWORK problem, not a credentials problem.")
            print("  Your keys were written to .env and may well be fine.")
            print("  Check: VPN or corporate proxy blocking paper-api.alpaca.markets,")
            print("  then re-run this script or go straight to preflight.py.")
        else:
            print("\n  Alpaca rejected the credentials. Most likely causes:")
            print("   - keys copied from a LIVE account rather than Paper Trading")
            print("   - the secret was truncated on paste")
            print("   - the key pair was regenerated, invalidating the old secret")
        return 1

    level = int(getattr(acct, "options_trading_level", 0) or 0)
    approved = int(getattr(acct, "options_approved_level", 0) or 0)

    print(f"  connected — account {acct.status}")
    print(f"  equity            ${float(acct.equity):,.2f}")
    print(f"  cash              ${float(acct.cash):,.2f}")
    print(f"  buying power      ${float(acct.buying_power):,.2f}")
    print(f"  options level     {level}  (approved: {approved})")

    print()
    if level >= 3:
        print("  LEVEL 3 — protective puts and multi-leg collars both available.")
    elif level == 2:
        print("  LEVEL 2 — long puts work; multi-leg collars do NOT.")
        print("  Request Level 3 in the dashboard if you want collars.")
    else:
        print(f"  LEVEL {level} — long puts are NOT available. The hedging leg")
        print("  cannot execute. Request an upgrade to Level 2 or 3 now:")
        print("  Alpaca dashboard -> Account -> Options Trading -> Apply.")
    return 0


def main() -> int:
    if ENV_PATH.exists():
        current = ENV_PATH.read_text(encoding="utf-8")
        looks_broken = "your_paper_key" in current or "read -" in current
        note = " (it looks like it contains placeholder or garbage values)" if looks_broken else ""
        answer = input(f".env already exists{note}. Overwrite? [y/N] ").strip().lower()
        if answer != "y":
            print("Cancelled — nothing written.")
            return 0

    api_key, secret, anthropic = prompt_keys()
    write_env(api_key, secret, anthropic)

    print(f"\n  ALPACA_API_KEY    = {mask(api_key)}")
    print(f"  ALPACA_SECRET_KEY = {mask(secret)}")
    if anthropic:
        print(f"  ANTHROPIC_API_KEY = {mask(anthropic)}")

    code = verify(api_key, secret)
    if code == 0:
        print("\nNext: PYTHONPATH=. python3 scripts/preflight.py 2>&1 | tee preflight.txt")
    return code


if __name__ == "__main__":
    raise SystemExit(main())
