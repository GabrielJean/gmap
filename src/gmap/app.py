"""Container entrypoint for the scraper + web dashboard."""

from __future__ import annotations

import signal
import sys
import threading
import time
from typing import Callable

from gmap.scraper import log_message, main as scrape_main
from gmap.web_app import app as flask_app


QUARTER_SECONDS = 900


def fmt_slot(ts: float) -> str:
    """Format an epoch timestamp in local time for logs."""
    return time.strftime("%Y-%m-%d %H:%M:%S %Z", time.localtime(ts))


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
    log_message("[app] Starting web dashboard on 0.0.0.0:5000")
    flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


def run_scraper_loop(stop_check: Callable[[], bool]) -> None:
    """Run scraper on quarter-hour slots; skip slots missed by long cycles."""
    cycle = 0

    now = time.time()
    next_slot = (int(now) // QUARTER_SECONDS) * QUARTER_SECONDS

    if next_slot <= now:
        delay = 0
    else:
        delay = next_slot - now

    log_message(
        "[app] Quarter-hour scheduler ready "
        f"(no slot discard); next slot {fmt_slot(next_slot)} "
        f"(in {int(delay)}s)"
    )

    while not stop_check():
        now = time.time()
        if now > next_slot:
            skipped_first_slot = next_slot
            skipped_slots = int((now - next_slot) // QUARTER_SECONDS) + 1
            next_slot = (int(now) // QUARTER_SECONDS + 1) * QUARTER_SECONDS
            log_message(
                f"[app] Previous cycle overran next slot; skipped {skipped_slots} slot(s) "
                f"starting at {fmt_slot(skipped_first_slot)}. Resuming at {fmt_slot(next_slot)}"
            )

        if now < next_slot:
            if not sleep_with_stop(next_slot - now, stop_check):
                return
            now = time.time()

        cycle += 1
        log_message(f"[app] Starting scrape cycle #{cycle} for slot {fmt_slot(next_slot)}")
        try:
            scrape_main()
            log_message(f"[app] Scrape cycle #{cycle} completed for slot {fmt_slot(next_slot)}")
        except Exception as exc:
            msg = f"[app] Scrape cycle #{cycle} failed with error {exc}; retrying in 60s"
            print(msg, file=sys.stderr)
            log_message(msg)
            if not sleep_with_stop(60, stop_check):
                return
        finally:
            next_slot += QUARTER_SECONDS

        if not stop_check() and time.time() < next_slot:
            log_message(
                f"[app] Next scrape scheduled at {fmt_slot(next_slot)} "
                f"(in {int(next_slot - time.time())}s)"
            )


def main() -> None:
    stop_event = threading.Event()

    def handle_signal(_signum: int, _frame) -> None:
        stop_event.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)
    log_message("[app] gmap service starting")

    web_thread = threading.Thread(target=run_web, name="web", daemon=True)
    web_thread.start()

    # Give the server a moment to bind before scraping.
    if not sleep_with_stop(5, stop_event.is_set):
        return

    run_scraper_loop(stop_event.is_set)


if __name__ == "__main__":
    main()
