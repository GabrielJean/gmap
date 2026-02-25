"""Scrape a list of URLs (dynamic pages too) with Playwright.

How to use with Chrome DevTools selectors:
1) Open the page in Chrome, right-click the element -> Inspect.
2) Right-click the highlighted node -> Copy -> Copy selector.
3) Paste that selector into the `selectors` dict below.


Set your URLs in `.env` (key `URLS`) as a JSON array or comma-separated list, then run:
	pip install playwright
	playwright install chromium
	PYTHONPATH=src python -m gmap.scraper

Outputs consistent fields to scrape_output.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import statistics
import time
from collections import defaultdict
from datetime import datetime, timezone, tzinfo, timedelta, date
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union
from urllib.parse import urlparse

import matplotlib
matplotlib.use("Agg")  # headless-safe backend for PNG output
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
from playwright.sync_api import Page, sync_playwright


HEADLESS = True  # set to False to watch the browser
OUTPUT_DIR = Path("data_runs")
LOG_PATH = OUTPUT_DIR / "logs.txt"
SNAPSHOT_LOG_PATH = OUTPUT_DIR / "snapshots.log"
ENV_PATH = Path(".env")
ENV_URL_KEY = "URLS"
def get_local_tz(key: str = "America/Toronto") -> tzinfo:
	"""Return desired TZ or UTC if tzdata is missing in the container."""
	try:
		return ZoneInfo(key)
	except ZoneInfoNotFoundError:
		print(f"Warning: timezone '{key}' not found; falling back to UTC")
		return timezone.utc


LOCAL_TZ = get_local_tz()

# Keep a stable field order for CSV-friendly outputs.
OUTPUT_FIELDS = [
	"timestamp_local",
	"day_of_week",
	"name",
	"page_title",
	"duration_minutes",
]

FRENCH_WEEKDAYS_FULL = [
	"Lundi",
	"Mardi",
	"Mercredi",
	"Jeudi",
	"Vendredi",
	"Samedi",
	"Dimanche",
]

FRENCH_WEEKDAYS_SHORT = [day[:3] for day in FRENCH_WEEKDAYS_FULL]

FRENCH_MONTH_ABBR = [
	"janv.",
	"févr.",
	"mars",
	"avr.",
	"mai",
	"juin",
	"juil.",
	"août",
	"sept.",
	"oct.",
	"nov.",
	"déc.",
]


@dataclass
class UrlEntry:
	name: str
	direction: str
	url: str

def load_env_file(env_path: Path = ENV_PATH) -> None:
	"""Populate os.environ with values from a simple .env file if present."""
	if not env_path.exists():
		return

	for line in env_path.read_text().splitlines():
		line = line.strip()
		if not line or line.startswith("#") or "=" not in line:
			continue
		key, value = line.split("=", 1)
		key = key.strip()
		# Remove surrounding quotes to make comma or JSON parsing easier later.
		value = value.strip().strip("\"").strip("'")
		os.environ.setdefault(key, value)


def parse_urls(raw_urls: str) -> List[str]:
	"""Return a list of URLs from a JSON array or comma-separated string."""
	urls: List[str] = []
	if not raw_urls:
		return urls

	try:
		parsed = json.loads(raw_urls)
		if isinstance(parsed, str):
			urls = [parsed]
		elif isinstance(parsed, list):
			urls = [u for u in parsed if isinstance(u, str)]
		else:
			parsed = []  # fall back to comma parsing below
	except json.JSONDecodeError:
		parsed = []

	if not urls and raw_urls:
		urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

	return [u for u in urls if u]


def parse_url_entries(raw_urls: str) -> List[UrlEntry]:
	"""Return structured URL entries from a JSON array of strings or objects."""
	if not raw_urls:
		return []
	try:
		parsed = json.loads(raw_urls)
	except json.JSONDecodeError:
		parsed = None

	entries: List[UrlEntry] = []
	if isinstance(parsed, list):
		for idx, item in enumerate(parsed):
			if isinstance(item, str):
				entries.append(UrlEntry(name=f"group-{idx + 1}", direction=f"dir-{idx + 1}", url=item))
			elif isinstance(item, dict):
				url = item.get("url") or item.get("URL") or item.get("href")
				if not isinstance(url, str) or not url.strip():
					continue
				name = item.get("name") or item.get("label") or f"group-{idx + 1}"
				direction = item.get("direction") or item.get("dir") or f"dir-{idx + 1}"
				entries.append(UrlEntry(name=name, direction=direction, url=url.strip()))
	elif isinstance(parsed, str):
		entries.append(UrlEntry(name="group-1", direction="dir-1", url=parsed))

	return entries


def load_urls(env_key: str = ENV_URL_KEY, env_path: Path = ENV_PATH) -> List[UrlEntry]:
	"""Load URLs from env var (JSON/CSV) with optional friendly names and directions."""
	load_env_file(env_path)

	raw = os.getenv(env_key, "")
	entries = parse_url_entries(raw)

	# Fallback: plain JSON/CSV list of URLs without objects
	if not entries:
		urls = parse_urls(raw)
		entries = [
			UrlEntry(name=f"group-{idx + 1}", direction=f"dir-{idx + 1}", url=url)
			for idx, url in enumerate(urls)
		]

	if not entries:
		raise SystemExit(
			f"Provide URLs via {env_key} in {env_path} (JSON array or comma-separated) before running."
		)
	return entries


def clean_segment(text: str) -> str:
	"""Make a string filesystem-safe (alnum, dash, underscore)."""
	cleaned = "".join(c if c.isalnum() or c in "-_" else "-" for c in text.strip())
	return cleaned or "default"


def log_message(message: str) -> None:
	"""Append a timestamped log line to the host-mounted logs file."""
	LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
	now = datetime.now(timezone.utc).astimezone(LOCAL_TZ).isoformat()
	with LOG_PATH.open("a", encoding="utf-8") as f:
		f.write(f"{now}\t{message}\n")


def log_snapshot(entry: UrlEntry, message: str) -> None:
	"""Append a timestamped line to a per-URL snapshot log file for easier navigation."""
	base = OUTPUT_DIR / "snapshots" / clean_segment(entry.name) / clean_segment(entry.direction)
	base.mkdir(parents=True, exist_ok=True)
	date_str = datetime.now(LOCAL_TZ).strftime("%Y-%m-%d")
	file_path = base / f"{date_str}.log"
	now = datetime.now(timezone.utc).astimezone(LOCAL_TZ).isoformat()
	with file_path.open("a", encoding="utf-8") as f:
		f.write(f"{now}\t{message}\n")


def safe_slug(entry: UrlEntry, index: int) -> str:
	"""Create a stable, filename-safe slug using friendly name, direction, and URL context."""
	parsed = urlparse(entry.url)
	host = (parsed.hostname or "url").replace(" ", "-")
	path_parts = [p.replace(" ", "-") for p in parsed.path.split("/") if p]
	path_part = "-".join(path_parts[:3]) if path_parts else "root"
	query_hint = ""
	if parsed.query:
		query_hint = f"-q{hashlib.sha1(parsed.query.encode('utf-8')).hexdigest()[:6]}"
	digest = hashlib.sha1(entry.url.encode("utf-8")).hexdigest()[:6]
	name_part = clean_segment(entry.name)
	dir_part = clean_segment(entry.direction)
	return f"{index + 1}-{name_part}-{dir_part}-{host}-{path_part}{query_hint}-{digest}"


def parse_duration_minutes(duration_text: Optional[str]) -> Optional[float]:
	"""Convert duration strings like '1 hr 5 min' or '45 min' to total minutes."""
	if not duration_text:
		return None

	normalized = duration_text.lower().replace("\xa0", " ")
	normalized = re.sub(r"\s+", " ", normalized).strip()

	hours = 0
	minutes = 0

	hour_match = re.search(r"(\d+)\s*(?:h|hr|hrs|heure|heures)\b", normalized, re.IGNORECASE)
	if hour_match:
		hours = int(hour_match.group(1))

	minute_match = re.search(r"(\d+)\s*(?:min|mins|minute|minutes)\b", normalized, re.IGNORECASE)
	if minute_match:
		minutes = int(minute_match.group(1))
	elif hour_match:
		# Handle compact forms like "1h05" or "1 h 05" where "min" is omitted.
		compact_minute_match = re.search(
			r"(?:\d+)\s*(?:h|hr|hrs|heure|heures)\s*(\d{1,2})\b",
			normalized,
			re.IGNORECASE,
		)
		if compact_minute_match:
			minutes = int(compact_minute_match.group(1))
	else:
		# Minute-only compact form like "45m".
		short_minute_match = re.search(r"(\d+)\s*m\b", normalized, re.IGNORECASE)
		if short_minute_match:
			minutes = int(short_minute_match.group(1))

	total_minutes = hours * 60 + minutes
	return total_minutes if total_minutes > 0 else None


def extract_duration_fragment(text: str) -> Optional[str]:
	"""Pull a plausible duration fragment (e.g., '1 hr 5 min' or '38 min') from free text."""
	patterns = [
		r"(\d+\s*(?:h|hr|hrs|heure|heures)\s*\d{1,2}\s*(?:min|mins|minute|minutes)?)",
		r"(\d+\s*(?:h|hr|hrs|heure|heures))",
		r"(\d+\s*(?:min|mins|minute|minutes|m))",
	]
	for pat in patterns:
		match = re.search(pat, text, re.IGNORECASE)
		if match:
			return match.group(1)
	return None


def _extract_duration_from_page_text(page: Page) -> Optional[str]:
	"""Ultimate fallback: regex the visible page text for a duration pattern.

	Collects ALL duration-like matches and returns the SHORTEST one.
	On a Google Maps directions page the driving time is almost always the
	shortest duration visible (compared to transit, walking, cycling, or
	total trip estimates).
	"""
	try:
		body = page.query_selector("body")
		if not body:
			return None
		full_text = (body.inner_text() or "")
		patterns = [
			r"(\d+\s*(?:h|hr|hrs|heure|heures)\s*\d{1,2}\s*(?:min|mins|minute|minutes)?)",
			r"(\d+\s*(?:min|mins|minute|minutes))",
		]
		candidates: List[Tuple[float, str]] = []
		for pat in patterns:
			for match in re.finditer(pat, full_text, re.IGNORECASE):
				candidate = match.group(1).strip()
				val = parse_duration_minutes(candidate)
				if val is not None and 1 <= val <= 600:
					candidates.append((val, candidate))
		if candidates:
			# Return the shortest plausible duration (most likely driving).
			candidates.sort(key=lambda c: c[0])
			return candidates[0][1]
	except Exception:
		pass
	return None


def _extract_duration_from_cards(page: Page) -> Optional[str]:
	"""Try extracting durations from route card elements, return shortest.

	Looks for trip-card elements by stable ID prefix and parses their
	text content for duration fragments.  Returns the shortest plausible
	driving duration found across all cards.
	"""
	nodes = page.query_selector_all("[id^='section-directions-trip-']")
	candidates: List[Tuple[float, str]] = []
	for node in nodes:
		text = (node.inner_text() or "").strip()
		if not text:
			continue
		flat = " ".join(text.split())
		frag = extract_duration_fragment(flat)
		if frag:
			val = parse_duration_minutes(frag)
			if val is not None and 1 <= val <= 600:
				candidates.append((val, frag))
	if candidates:
		candidates.sort(key=lambda c: c[0])
		return candidates[0][1]
	return None


def extract_duration(page: Page) -> Tuple[Optional[float], Optional[str], str]:
	"""Extract driving duration from the current page.

	Returns (duration_minutes, raw_text, method) where *method* indicates
	which extraction strategy succeeded.

	Strategy order (most targeted first):
	  1. Route card text – parse trip-card elements by ID prefix.
	  2. Full-page regex – scan all visible text, pick shortest match.
	"""
	# 1. Route cards (targeted, less noise).
	card_frag = _extract_duration_from_cards(page)
	if card_frag:
		val = parse_duration_minutes(card_frag)
		if val is not None:
			return val, card_frag, "card-text"

	# 2. Full-page regex (reliable even when DOM classes change).
	page_frag = _extract_duration_from_page_text(page)
	if page_frag:
		val = parse_duration_minutes(page_frag)
		if val is not None:
			return val, page_frag, "page-regex"

	return None, None, "none"


def snapshot_text(page: Page) -> str:
	"""Capture a lightweight text snapshot from the first route card (fallback to body)."""
	for sel in ["#section-directions-trip-0", "body"]:
		node = page.query_selector(sel)
		if node:
			text = (node.inner_text() or "").strip()
			if text:
				flat = " ".join(text.split())
				return flat[:800]
	return ""


def _save_debug_screenshot(page: Page, entry: "UrlEntry") -> Optional[Path]:
	"""Save a screenshot for debugging when duration extraction fails."""
	try:
		screenshots_dir = OUTPUT_DIR / "debug_screenshots"
		screenshots_dir.mkdir(parents=True, exist_ok=True)
		name_part = clean_segment(entry.name)
		dir_part = clean_segment(entry.direction)
		ts_str = datetime.now(LOCAL_TZ).strftime("%Y%m%d-%H%M%S")
		path = screenshots_dir / f"{name_part}-{dir_part}-{ts_str}.png"
		page.screenshot(path=str(path), full_page=False)
		# Only keep the 20 most recent screenshots to avoid filling disk.
		all_shots = sorted(screenshots_dir.glob("*.png"), key=lambda p: p.stat().st_mtime)
		for old in all_shots[:-20]:
			old.unlink(missing_ok=True)
		return path
	except Exception:
		return None


def write_url_manifest(entries: List[UrlEntry], slugs: List[str], timestamp: str) -> None:
	"""Deprecated: per-run manifests removed in favor of per-URL histories."""
	raise NotImplementedError("Manifest writing is disabled; use per-URL CSVs instead.")



def write_outputs_for_result(result: Dict[str, Optional[Any]], entry: UrlEntry, slug: str) -> None:
	"""Append per-URL CSV (single rolling file) grouped by name/direction."""
	name_part = clean_segment(entry.name)
	dir_part = clean_segment(entry.direction)
	output_base = OUTPUT_DIR / name_part / dir_part
	output_base.mkdir(parents=True, exist_ok=True)

	# Keep file names short: just the friendly name + direction
	csv_path = output_base / f"{name_part}-{dir_part}.csv"
	write_header = not csv_path.exists()
	with csv_path.open("a", newline="") as f:
		writer = csv.DictWriter(f, fieldnames=OUTPUT_FIELDS)
		if write_header:
			writer.writeheader()
		writer.writerow({field: result.get(field) for field in OUTPUT_FIELDS})


def parse_timestamp(value: str) -> Optional[datetime]:
	"""Parse ISO timestamps while tolerating missing timezone info."""
	if not value:
		return None
	try:
		dt = datetime.fromisoformat(value)
		dt = dt.astimezone(LOCAL_TZ) if dt.tzinfo else dt.replace(tzinfo=LOCAL_TZ)
		return dt
	except Exception:
		return None


def load_csv_points(csv_path: Path) -> List[Tuple[datetime, float]]:
	"""Return sorted (timestamp, duration_minutes) tuples for plotting."""
	points: List[Tuple[datetime, float]] = []
	with csv_path.open() as f:
		reader = csv.DictReader(f)
		for row in reader:
			ts = parse_timestamp(row.get("timestamp_local", ""))
			try:
				duration = float(row.get("duration_minutes", ""))
			except (TypeError, ValueError):
				duration = None
			if ts and duration is not None:
				points.append((ts, duration))
	return sorted(points, key=lambda p: p[0])


def is_peak(ts: datetime) -> bool:
	"""Return True if timestamp is within a defined peak bucket."""
	hour = ts.astimezone(LOCAL_TZ).hour
	return any(in_range(hour, start, end) for _, start, end in PEAK_BUCKETS)


def weekday_peak_medians(points: List[Tuple[datetime, float]]) -> Dict[int, float]:
	"""Compute median duration per weekday using any peak sample (AM or PM)."""
	by_day: Dict[int, List[float]] = defaultdict(list)
	for ts, duration in points:
		if is_peak(ts):
			by_day[ts.weekday()].append(duration)
	return {day: statistics.median(values) for day, values in by_day.items() if values}


def weekday_peak_bucket_medians(points: List[Tuple[datetime, float]]) -> Dict[int, Dict[str, float]]:
	"""Compute median per weekday, split by each peak bucket (e.g., AM/PM)."""
	by_day_bucket: Dict[int, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
	for ts, duration in points:
		hour = ts.astimezone(LOCAL_TZ).hour
		for name, start, end in PEAK_BUCKETS:
			if in_range(hour, start, end):
				by_day_bucket[ts.weekday()][name].append(duration)
	result: Dict[int, Dict[str, float]] = {}
	for day, bucket_map in by_day_bucket.items():
		result[day] = {bucket: statistics.median(vals) for bucket, vals in bucket_map.items() if vals}
	return result


def daily_median_series(points: List[Tuple[datetime, float]]) -> List[Tuple[datetime, float]]:
	"""Collapse to one median point per calendar day for long-range readability."""
	by_date: Dict[date, List[float]] = defaultdict(list)
	for ts, duration in points:
		by_date[ts.date()].append(duration)
	series: List[Tuple[datetime, float]] = []
	for day, values in by_date.items():
		series.append((datetime.combine(day, datetime.min.time(), tzinfo=LOCAL_TZ), statistics.median(values)))
	return sorted(series, key=lambda p: p[0])


def weekly_median_series(points: List[Tuple[datetime, float]]) -> List[Tuple[datetime, float]]:
	"""Collapse to one median point per ISO week for multi-month readability."""
	by_week: Dict[Tuple[int, int], List[float]] = defaultdict(list)
	week_start: Dict[Tuple[int, int], datetime] = {}
	for ts, duration in points:
		iso_year, iso_week, iso_weekday = ts.isocalendar()
		key = (iso_year, iso_week)
		by_week[key].append(duration)
		if key not in week_start:
			start = (ts - timedelta(days=iso_weekday - 1)).date()
			week_start[key] = datetime.combine(start, datetime.min.time(), tzinfo=LOCAL_TZ)
	series: List[Tuple[datetime, float]] = []
	for key, values in by_week.items():
		series.append((week_start[key], statistics.median(values)))
	return sorted(series, key=lambda p: p[0])


# Define peak buckets (local time, 24h). Adjust as needed.
PEAK_BUCKETS: List[Tuple[str, int, int]] = [
	("Pic matin", 7, 9),   # 07:00-09:00
	("Pic après-midi", 15, 17), # 15:00-17:00
]


def in_range(hour: int, start: int, end: int) -> bool:
	"""Return True if hour is within [start, end) handling wrap at midnight."""
	if start <= end:
		return start <= hour < end
	return hour >= start or hour < end


def bucket_medians(points: List[Tuple[datetime, float]], buckets: List[Tuple[str, int, int]]) -> Dict[str, float]:
	"""Median per time-of-day bucket for provided ranges."""
	by_bucket: Dict[str, List[float]] = defaultdict(list)
	for ts, duration in points:
		hour = ts.astimezone(LOCAL_TZ).hour
		for name, start, end in buckets:
			if in_range(hour, start, end):
				by_bucket[name].append(duration)
	return {name: statistics.median(vals) for name, vals in by_bucket.items() if vals}


def offpeak_median(points: List[Tuple[datetime, float]]) -> Optional[float]:
	"""Median for samples outside peak windows."""
	values: List[float] = []
	for ts, duration in points:
		if not is_peak(ts):
			values.append(duration)
	return statistics.median(values) if values else None


def percentile(values: List[float], pct: float) -> Optional[float]:
	"""Return percentile (0-100) using linear interpolation."""
	if not values:
		return None
	sorted_vals = sorted(values)
	if pct <= 0:
		return sorted_vals[0]
	if pct >= 100:
		return sorted_vals[-1]
	k = (len(sorted_vals) - 1) * (pct / 100)
	f = int(k)
	c = min(f + 1, len(sorted_vals) - 1)
	if f == c:
		return sorted_vals[f]
	d0 = sorted_vals[f] * (c - k)
	d1 = sorted_vals[c] * (k - f)
	return d0 + d1


def daily_percentile_band(
	points: List[Tuple[datetime, float]],
	low_pct: float,
	high_pct: float,
) -> Tuple[List[datetime], List[float], List[float]]:
	"""Return daily percentile bands (low/high) for each day."""
	by_date: Dict[date, List[float]] = defaultdict(list)
	for ts, duration in points:
		by_date[ts.date()].append(duration)
	dates: List[datetime] = []
	low_vals: List[float] = []
	high_vals: List[float] = []
	for day in sorted(by_date.keys()):
		vals = by_date[day]
		low = percentile(vals, low_pct)
		high = percentile(vals, high_pct)
		if low is None or high is None:
			continue
		dates.append(datetime.combine(day, datetime.min.time(), tzinfo=LOCAL_TZ))
		low_vals.append(low)
		high_vals.append(high)
	return dates, low_vals, high_vals


def format_date_fr(x: float, _: Optional[int] = None) -> str:
	"""Return a French day-month label regardless of system locale."""
	dt = mdates.num2date(x, tz=LOCAL_TZ)
	month_label = FRENCH_MONTH_ABBR[dt.month - 1]
	return f"{dt.day:02d} {month_label}"


def generate_graph(csv_path: Path) -> Optional[Path]:
	"""Create/overwrite a PNG for the given CSV; returns image path."""
	points = load_csv_points(csv_path)
	if not points:
		return None
	weekday_peaks = weekday_peak_medians(points)
	weekday_peak_buckets = weekday_peak_bucket_medians(points)
	daily = daily_median_series(points)
	weekly = weekly_median_series(points)
	band_dates, band_low, band_high = daily_percentile_band(points, 5, 95)
	peak_buckets = bucket_medians(points, PEAK_BUCKETS)
	offpeak_value = offpeak_median(points)

	x_vals, y_vals = zip(*points)
	img_path = csv_path.with_suffix(".png")

	# Build a rolling median to reduce noise (window ~ 2 days or 20 samples min)
	roll_points: List[Tuple[datetime, float]] = []
	window = max(20, min(240, len(points) // 12))
	for idx in range(len(points)):
		start = max(0, idx - window + 1)
		chunk = [p[1] for p in points[start : idx + 1]]
		if chunk:
			roll_points.append((points[idx][0], statistics.median(chunk)))

	# Decimate raw samples for long ranges (keep trends without overplotting)
	max_points = 5000
	if len(points) > max_points:
		step = max(1, len(points) // max_points)
		decimated = points[::step]
	else:
		decimated = points
	x_vals, y_vals = zip(*decimated)

	fig = plt.figure(figsize=(13.2, 11.2))
	gs = fig.add_gridspec(
		4,
		2,
		width_ratios=[7, 2.2],
		height_ratios=[3, 2, 2, 2],
		wspace=0.35,
		hspace=0.38,
	)
	ax_raw = fig.add_subplot(gs[0, 0])
	ax_trend = fig.add_subplot(gs[1, 0], sharex=ax_raw)
	ax_band = fig.add_subplot(gs[2, 0], sharex=ax_raw)
	ax_week = fig.add_subplot(gs[3, 0])
	stats_ax = fig.add_subplot(gs[:, 1])
	stats_ax.axis("off")
	fig.suptitle(csv_path.stem.replace("-", " "))
	min_dt = points[0][0]
	max_dt = points[-1][0]
	range_label = f"Période : {min_dt.strftime('%Y-%m-%d')} → {max_dt.strftime('%Y-%m-%d')}"
	fig.text(0.02, 0.975, range_label, ha="left", va="top", fontsize=9)

	primary_color = "#1f77b4"
	trend_color = "#d62728"
	roll_color = "#2ca02c"
	weekly_color = "#9467bd"

	# Panel 1: raw samples + extremes
	ax_raw.plot(x_vals, y_vals, linewidth=1.0, color=primary_color, alpha=0.35, label="Échantillons (décimés)")
	values = [p[1] for p in points]
	p98 = percentile(values, 98)
	if p98 is not None:
		extreme_points = [(ts, val) for ts, val in points if val >= p98]
		if extreme_points:
			ex, ey = zip(*extreme_points)
			ax_raw.scatter(ex, ey, s=14, color="#e41a1c", alpha=0.75, label="Valeurs extrêmes")
	ax_raw.set_ylabel("Minutes")
	ax_raw.grid(True, linewidth=0.5, alpha=0.25)
	ax_raw.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), framealpha=0.9, ncol=2)

	# Panel 2: trend lines only
	if daily:
		dx, dy = zip(*daily)
		ax_trend.plot(dx, dy, linewidth=2.1, color=trend_color, alpha=0.9, label="Médiane quotidienne")
	if weekly:
		wx, wy = zip(*weekly)
		ax_trend.plot(wx, wy, linewidth=2.3, color=weekly_color, alpha=0.9, label="Médiane hebdomadaire")
	if roll_points:
		rx, ry = zip(*roll_points)
		ax_trend.plot(rx, ry, linewidth=1.6, color=roll_color, alpha=0.9, label="Médiane glissante")
	all_median = statistics.median(values)
	ax_trend.axhline(all_median, color="#555555", linewidth=0.8, linestyle="--", alpha=0.6)
	if offpeak_value is not None:
		ax_trend.axhline(offpeak_value, color="#999999", linewidth=0.8, linestyle=":", alpha=0.6)
	ax_trend.set_ylabel("Tendance (min)")
	ax_trend.grid(True, linewidth=0.5, alpha=0.25)
	ax_trend.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), framealpha=0.9, ncol=3)

	# Panel 3: daily percentile band
	if band_dates:
		ax_band.fill_between(
			band_dates,
			band_low,  # type: ignore[arg-type]
			band_high,  # type: ignore[arg-type]
			color="#bdbdbd",
			alpha=0.35,
			label="Bande quotidienne 5–95 %",
		)
	if daily:
		dx, dy = zip(*daily)
		ax_band.plot(dx, dy, linewidth=1.6, color=trend_color, alpha=0.9, label="Médiane quotidienne")
	ax_band.set_ylabel("Bande quotidienne")
	ax_band.grid(True, linewidth=0.5, alpha=0.25)
	ax_band.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), framealpha=0.9, ncol=2)

	locator = mdates.AutoDateLocator(minticks=6, maxticks=10)
	formatter = FuncFormatter(format_date_fr)
	for axis in (ax_raw, ax_trend, ax_band):
		axis.xaxis.set_major_locator(locator)
		axis.xaxis.set_major_formatter(formatter)
		axis.label_outer()  # type: ignore[attr-defined]

	# Panel 4: weekday peak medians (AM/PM bars) or single peak median
	weekday_order = list(range(7))
	weekday_labels = FRENCH_WEEKDAYS_SHORT
	indices = list(range(len(weekday_order)))
	morning_label, afternoon_label = PEAK_BUCKETS[0][0], PEAK_BUCKETS[1][0]

	if weekday_peak_buckets:
		am_vals = [weekday_peak_buckets.get(day, {}).get(morning_label) for day in weekday_order]
		pm_vals = [weekday_peak_buckets.get(day, {}).get(afternoon_label) for day in weekday_order]
		width = 0.38
		ax_week.bar(
			[i - width / 2 for i in indices],
			[v if v is not None else 0 for v in am_vals],
			width=width,
			color="#8da0cb",
			label=morning_label,
		)
		ax_week.bar(
			[i + width / 2 for i in indices],
			[v if v is not None else 0 for v in pm_vals],
			width=width,
			color="#fc8d62",
			label=afternoon_label,
		)
		ax_week.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), framealpha=0.9, ncol=2)
	elif weekday_peaks:
		vals = [weekday_peaks.get(day) for day in weekday_order]
		ax_week.bar(indices, [v if v is not None else 0 for v in vals], color="#9ecae1", label="Médiane du pic")
		ax_week.legend(loc="upper center", bbox_to_anchor=(0.5, -0.12), framealpha=0.9)

	ax_week.set_xticks(indices)
	ax_week.set_xticklabels(weekday_labels)
	ax_week.set_ylabel("Pic (min)")
	ax_week.grid(True, axis="y", linewidth=0.5, alpha=0.25)

	# Compact stats panel (dedicated column)
	stats_blocks: List[str] = []
	if peak_buckets:
		peak_lines = [f"{name}: {val:.1f}" for name, val in sorted(peak_buckets.items())]
		stats_blocks.append("Médianes des pics\n" + "\n".join(peak_lines))
	stats_blocks.append(f"Médiane globale : {all_median:.1f}")
	p95 = percentile(values, 95)
	max_val = max(values) if values else None
	if p95 is not None:
		stats_blocks.append(f"P95 : {p95:.1f}")
	if max_val is not None:
		stats_blocks.append(f"Max : {max_val:.1f}")
	if offpeak_value is not None:
		stats_blocks.append(f"Médiane hors pointe : {offpeak_value:.1f}")
	if stats_blocks:
		stats_ax.text(
			0.0,
			1.0,
			"\n\n".join(stats_blocks),
			va="top",
			ha="left",
			fontsize=8,
			bbox={"facecolor": "white", "alpha": 0.9, "edgecolor": "none"},
			transform=stats_ax.transAxes,
		)

	interpretation = (
		"Comment lire : (1) Échantillons = un résultat toutes les 15 min ; points rouges = délais anormalement longs. "
		"(2) Tendances = médianes quotid./hebdo./glissante ; la médiane glissante montre la tendance à court terme.\n"
		"(3) Bande quotidienne = plage habituelle de la journée (la plupart des valeurs sont dedans). "
		"(4) Barres par jour = pics matin/après-midi typiques."
	)
	fig.text(0.02, 0.035, interpretation, ha="left", va="bottom", fontsize=8)

	fig.subplots_adjust(left=0.06, bottom=0.16, right=0.78, top=0.96)
	img_path.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(str(img_path), dpi=140)
	plt.close(fig)
	return img_path


def generate_all_graphs(output_dir: Path = OUTPUT_DIR) -> None:
	"""Generate/refresh PNG graphs for every per-URL CSV."""
	csv_paths: List[Path] = sorted(output_dir.glob("*/*/*.csv"))
	log_message(f"[graphs] Starting graph generation for {len(csv_paths)} CSV file(s)")
	for i, csv_path in enumerate(csv_paths, 1):
		try:
			generate_graph(csv_path)
			log_message(f"[graphs] {i}/{len(csv_paths)} generated: {csv_path.stem}")
		except Exception as exc:
			log_message(f"[graphs] {i}/{len(csv_paths)} FAILED for {csv_path}: {exc}")
	log_message(f"[graphs] Graph generation complete ({len(csv_paths)} file(s))")


def main() -> None:
	urls = load_urls()
	cycle_started = time.monotonic()
	log_message(f"[scraper] Cycle start; {len(urls)} route(s) configured")

	results: List[Dict[str, Optional[Any]]] = []
	missing_duration_count = 0
	now_utc = datetime.now(timezone.utc)
	local_dt = now_utc.astimezone(LOCAL_TZ)
	timestamp_local = local_dt.isoformat()
	day_of_week = local_dt.strftime("%A")

	with sync_playwright() as p:
		browser = p.chromium.launch(headless=HEADLESS)
		page = browser.new_page()

		# Set a common UA to reduce bot friction and extend default timeouts
		page.set_extra_http_headers({
			"User-Agent": (
				"Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
				"AppleWebKit/537.36 (KHTML, like Gecko) "
				"Chrome/120.0 Safari/537.36"
			)
		})
		page.set_default_timeout(60_000)

		for idx, entry in enumerate(urls):
			slug = safe_slug(entry, idx)
			log_message(
				f"[scraper] Route {idx + 1}/{len(urls)} start: '{entry.name}' ({entry.direction})"
			)

			# --- Navigate ---
			try:
				page.goto(entry.url, wait_until="load", timeout=60_000)
				try:
					page.wait_for_selector(
						"#section-directions-trip-0, #section-directions-trip-1, "
						"[data-trip-index], [class*='directions-trip']",
						timeout=30_000,
					)
				except Exception:
					pass
			except Exception as exc:
				log_message(
					f"[scraper] Route {idx + 1}/{len(urls)} failed: '{entry.name}' ({entry.direction}) "
					f"url={entry.url} error={exc}; continuing with remaining routes"
				)
				continue

			# --- Get page title ---
			title_node = page.query_selector("title")
			page_title = (title_node.inner_text().strip()) if title_node else None

			# --- Extract duration ---
			duration_minutes, duration_text, method = extract_duration(page)

			if duration_minutes is None:
				missing_duration_count += 1
				snap = snapshot_text(page)
				shot = _save_debug_screenshot(page, entry)
				log_snapshot(
					entry,
					f"Missing duration for '{entry.name}' ({entry.direction}) url={entry.url}; "
					f"screenshot={shot} snapshot='{snap}'"
				)
				log_message(
					f"[scraper] Route {idx + 1}/{len(urls)} done: '{entry.name}' ({entry.direction}) "
					f"duration=missing (screenshot={shot})"
				)
			else:
				log_message(
					f"[scraper] Route {idx + 1}/{len(urls)} done: '{entry.name}' ({entry.direction}) "
					f"duration={duration_minutes:.1f} min (method={method}, raw='{duration_text}')"
				)

			result: Dict[str, Optional[Any]] = {
				"timestamp_local": timestamp_local,
				"day_of_week": day_of_week,
				"name": entry.name,
				"page_title": page_title,
				"duration_minutes": duration_minutes,
			}
			write_outputs_for_result(result, entry, slug)
			results.append(result)

		browser.close()

	generate_all_graphs()
	elapsed = time.monotonic() - cycle_started
	log_message(
		f"[scraper] Cycle complete; wrote {len(results)} row(s), missing_duration={missing_duration_count}, "
		f"elapsed={elapsed:.1f}s"
	)

	print(f"Wrote {len(results)} rows (per-URL CSV histories only)")


if __name__ == "__main__":
	main()
