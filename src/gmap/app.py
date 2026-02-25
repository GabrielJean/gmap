"""Container entrypoint for the scraper + web dashboard."""

from __future__ import annotations

import signal
import sys
import threading
import time
import traceback
from typing import Callable, Optional

from gmap.scraper import log_message, main as scrape_main
from gmap.web_app import app as flask_app


QUARTER_SECONDS = 900
# Maximum time (seconds) a single scrape cycle may run before being abandoned.
# Slightly under one quarter-hour so we never silently miss the next slot.
MAX_CYCLE_SECONDS = 13 * 60  # 13 minutes


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


def _run_scrape_with_timeout(timeout: float) -> Optional[str]:
    """Run scrape_main() in a daemon thread with a hard timeout.

    Returns None on success, or an error message string on failure/timeout.
    """
    result: Optional[str] = None

    def _target() -> None:
        nonlocal result
        try:
            scrape_main()
        except Exception as exc:
            result = f"{exc}\n{traceback.format_exc()}"

    worker = threading.Thread(target=_target, name="scrape-worker", daemon=True)
    worker.start()
    worker.join(timeout=timeout)
    if worker.is_alive():
        return (
            f"Scrape cycle timed out after {int(timeout)}s — the worker thread will be "
            "abandoned (daemon) and cleaned up on next cycle or process restart"
        )
    return result


def run_web() -> None:
    """Run the Flask dashboard server."""
    log_message("[app] Starting web dashboard on 0.0.0.0:5000")
    flask_app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)


def run_scraper_loop(stop_check: Callable[[], bool]) -> None:
    """Run scraper on quarter-hour slots; skip slots missed by long cycles.

    Resilience guarantees
    ---------------------
    * First scrape runs immediately on startup (no wasted quarter-hour wait).
    * Each cycle is wrapped in a daemon-thread with a hard timeout so a hung
      Playwright session can never block the scheduler permanently.
    * All exceptions (including BaseException subclasses other than
      SystemExit / KeyboardInterrupt) are caught and logged.
    * Consecutive failures are counted and logged so persistent problems are
      visible in the log file.
    """
    cycle = 0
    consecutive_failures = 0

    # ── First scrape: run immediately at the current (truncated) slot ──
    now = time.time()
    current_slot = (int(now) // QUARTER_SECONDS) * QUARTER_SECONDS
    next_slot = current_slot  # will be executed right away

    log_message(
        f"[app] Quarter-hour scheduler ready; first scrape immediately "
        f"at slot {fmt_slot(next_slot)}"
    )

    while not stop_check():
        now = time.time()

        # If we're past the target slot (e.g. after a long cycle), fast-forward
        # to the next future quarter-hour and log what was skipped.
        if now > next_slot + QUARTER_SECONDS:
            skipped_first_slot = next_slot
            skipped_slots = int((now - next_slot) // QUARTER_SECONDS)
            next_slot = (int(now) // QUARTER_SECONDS) * QUARTER_SECONDS
            log_message(
                f"[app] Previous cycle overran; skipped {skipped_slots} slot(s) "
                f"starting at {fmt_slot(skipped_first_slot)}. "
                f"Resuming at {fmt_slot(next_slot)}"
            )

        # Sleep until the target slot (may be zero on first iteration).
        if now < next_slot:
            if not sleep_with_stop(next_slot - now, stop_check):
                return

        cycle += 1
        log_message(f"[app] Starting scrape cycle #{cycle} for slot {fmt_slot(next_slot)}")

        try:
            error = _run_scrape_with_timeout(MAX_CYCLE_SECONDS)
            if error is None:
                log_message(
                    f"[app] Scrape cycle #{cycle} completed for slot {fmt_slot(next_slot)}"
                )
                consecutive_failures = 0
            else:
                consecutive_failures += 1
                msg = (
                    f"[app] Scrape cycle #{cycle} failed "
                    f"(consecutive_failures={consecutive_failures}): {error}"
                )
                print(msg, file=sys.stderr)
                log_message(msg)
        except BaseException as exc:
            # Catch truly unexpected errors (e.g. interpreter-level) so the
            # scheduler never dies while the container stays up.
            if isinstance(exc, (SystemExit, KeyboardInterrupt)):
                raise
            consecutive_failures += 1
            msg = (
                f"[app] Scrape cycle #{cycle} hit unexpected error "
                f"(consecutive_failures={consecutive_failures}): "
                f"{exc}\n{traceback.format_exc()}"
            )
            print(msg, file=sys.stderr)
            log_message(msg)

        # Advance to the next quarter-hour slot.
        next_slot = (int(time.time()) // QUARTER_SECONDS + 1) * QUARTER_SECONDS

        if not stop_check():
            wait_secs = int(next_slot - time.time())
            log_message(
                f"[app] Next scrape scheduled at {fmt_slot(next_slot)} "
                f"(in {wait_secs}s)"
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
