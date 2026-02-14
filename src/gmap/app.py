"""Container entrypoint for the scraper + web dashboard."""

from __future__ import annotations

import signal
import sys
import threading
import time
from typing import Callable

from gmap.scraper import main as scrape_main
from gmap.web_app import app as flask_app


QUARTER_SECONDS = 900


def sleep_until_next_slot(stop_check: Callable[[], bool]) -> bool:
    """Sleep until the next quarter-hour slot; return False if stopped."""
    now = time.time()
    next_slot = (int(now) // QUARTER_SECONDS + 1) * QUARTER_SECONDS
    delay = max(0, next_slot - now)
    return sleep_with_stop(delay, stop_check)


def sleep_with_stop(delay: float, stop_check: Callable[[], bool]) -> bool:
    """Sleep in small chunks so we can honor stop requests."""
    remaining = delay
    while remaining > 0:
        if stop_check():
            return False
        chunk = min(1.0, remaining)
        time.sleep(chunk)
        remaining -= chunk
    return not stop_check()


def run_web() -> None:
    """Run the Flask dashboard server."""
    flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


def run_scraper_loop(stop_check: Callable[[], bool]) -> None:
    """Run one scrape immediately, then continue on quarter-hour cadence."""
    while not stop_check():
        try:
            scrape_main()
        except Exception as exc:
            print(f"gmap run failed with error {exc}; retrying in 60s", file=sys.stderr)
            if not sleep_with_stop(60, stop_check):
                return
            continue

        if not sleep_until_next_slot(stop_check):
            return


def main() -> None:
    stop_event = threading.Event()

    def handle_signal(_signum: int, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    web_thread = threading.Thread(target=run_web, name="web", daemon=True)
    web_thread.start()

    # Give the server a moment to bind before scraping.
    if not sleep_with_stop(5, stop_event.is_set):
        return

    run_scraper_loop(stop_event.is_set)


if __name__ == "__main__":
    main()
