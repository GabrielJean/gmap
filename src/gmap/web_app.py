"""Web dashboard for viewing gmap traffic data."""

from __future__ import annotations

import json
import csv
import io
import os
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from flask import Flask, render_template, jsonify, send_file, abort, request
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


app = Flask(__name__)

OUTPUT_DIR = Path("data_runs")


def get_local_tz(key: str = "America/Toronto"):
    """Return desired TZ or UTC if tzdata is missing."""
    try:
        return ZoneInfo(key)
    except ZoneInfoNotFoundError:
        from datetime import timezone
        return timezone.utc


LOCAL_TZ = get_local_tz()


def scan_data_structure() -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Scan data_runs directory and return structure of available data."""
    structure = {}

    if not OUTPUT_DIR.exists():
        return structure

    # Scan name directories
    for name_dir in OUTPUT_DIR.iterdir():
        if not name_dir.is_dir() or name_dir.name in ['logs.txt', 'snapshots', 'snapshots.log']:
            continue

        name = name_dir.name
        structure[name] = {}

        # Scan direction directories
        for dir_dir in name_dir.iterdir():
            if not dir_dir.is_dir():
                continue

            direction = dir_dir.name

            # Find CSV and PNG files
            csv_file = None
            png_file = None
            for file in dir_dir.iterdir():
                if file.suffix == '.csv':
                    csv_file = file
                elif file.suffix == '.png':
                    png_file = file

            if csv_file:
                # Read CSV metadata
                row_count = 0
                latest_timestamp = None
                earliest_timestamp = None

                try:
                    with csv_file.open() as f:
                        reader = csv.DictReader(f)
                        rows = list(reader)
                        row_count = len(rows)

                        if rows:
                            earliest_timestamp = rows[0].get('timestamp_local')
                            latest_timestamp = rows[-1].get('timestamp_local')
                except Exception:
                    pass

                structure[name][direction] = {
                    'csv_path': str(csv_file.relative_to(OUTPUT_DIR)),
                    'png_path': str(png_file.relative_to(OUTPUT_DIR)) if png_file else None,
                    'row_count': row_count,
                    'earliest_timestamp': earliest_timestamp,
                    'latest_timestamp': latest_timestamp,
                }

    return structure


def read_csv_data(csv_path: Path, limit: int = 1000) -> List[Dict[str, Any]]:
    """Read CSV file and return data as list of dicts."""
    data = []

    try:
        with csv_path.open() as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            # Return most recent rows if limit exceeded
            if len(rows) > limit:
                rows = rows[-limit:]

            for row in rows:
                data.append(row)
    except Exception as e:
        print(f"Error reading CSV {csv_path}: {e}")

    return data


@app.route('/')
def index():
    """Main dashboard page."""
    structure = scan_data_structure()
    return render_template('index.html', structure=structure)


@app.route('/route/<path:name>/<path:direction>')
def route_detail(name: str, direction: str):
    """Route detail page with focused stats."""
    structure = scan_data_structure()
    route_data = structure.get(name, {}).get(direction)

    if not route_data:
        abort(404)

    return render_template(
        'route.html',
        name=name,
        direction=direction,
        data=route_data,
    )


@app.route('/api/structure')
def api_structure():
    """Return data structure as JSON."""
    structure = scan_data_structure()
    return jsonify(structure)


@app.route('/api/data/<path:csv_path>')
def api_data(csv_path: str):
    """Return CSV data as JSON."""
    full_path = OUTPUT_DIR / csv_path

    if not full_path.exists() or not full_path.is_file():
        abort(404)

    # Security check - ensure path is within OUTPUT_DIR
    try:
        full_path.resolve().relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        abort(403)

    limit = 10000  # Maximum rows to return
    data = read_csv_data(full_path, limit=limit)

    return jsonify(data)


@app.route('/image/<path:png_path>')
def serve_image(png_path: str):
    """Serve PNG image."""
    full_path = OUTPUT_DIR / png_path

    if not full_path.exists() or not full_path.is_file():
        abort(404)

    # Security check
    try:
        full_path.resolve().relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        abort(403)

    return send_file(full_path, mimetype='image/png')


@app.route('/api/config')
def api_config():
    """Return configured URLs from .env file."""
    try:
        # Load URLs from .env (same method used by scraper)
        env_path = Path(".env")
        if not env_path.exists():
            return jsonify({"error": "Configuration file not found"}), 404

        # Load .env file
        for line in env_path.read_text().splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)

        # Parse URLs
        raw_urls = os.getenv("URLS", "")

        if not raw_urls:
            return jsonify({"urls": [], "raw": ""})

        # Try to parse as JSON
        try:
            parsed = json.loads(raw_urls)
            urls = parsed if isinstance(parsed, list) else [parsed] if isinstance(parsed, str) else []
        except json.JSONDecodeError:
            # Fall back to comma-separated
            urls = [u.strip() for u in raw_urls.split(",") if u.strip()]

        return jsonify({
            "urls": urls,
            "count": len(urls)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/logs')
def api_logs():
    """Return recent log entries."""
    log_path = OUTPUT_DIR / 'logs.txt'

    if not log_path.exists():
        return jsonify([])

    logs = []
    try:
        with log_path.open() as f:
            lines = f.readlines()
            # Return last 100 log lines
            for line in lines[-100:]:
                logs.append(line.strip())
    except Exception as e:
        print(f"Error reading logs: {e}")

    return jsonify(logs)


def _resolve_csv_path(csv_path: str) -> Path:
    """Resolve and validate a CSV path relative to OUTPUT_DIR."""
    full_path = OUTPUT_DIR / csv_path
    if not full_path.exists() or not full_path.is_file():
        abort(404)
    try:
        full_path.resolve().relative_to(OUTPUT_DIR.resolve())
    except ValueError:
        abort(403)
    if full_path.suffix != '.csv':
        abort(400)
    return full_path


@app.route('/api/csv/raw/<path:csv_path>')
def api_csv_raw(csv_path: str):
    """Return ALL rows from a CSV as JSON (no limit) for the editor."""
    full_path = _resolve_csv_path(csv_path)
    data = read_csv_data(full_path, limit=999_999)
    # Also return the column order from the header
    headers: List[str] = []
    try:
        with full_path.open() as f:
            reader = csv.reader(f)
            headers = next(reader, [])
    except Exception:
        pass
    return jsonify({"headers": headers, "rows": data})


@app.route('/api/csv/save/<path:csv_path>', methods=['POST'])
def api_csv_save(csv_path: str):
    """Save edited CSV data.  Expects JSON body: {headers: [...], rows: [{...}, ...]}."""
    full_path = _resolve_csv_path(csv_path)

    body = request.get_json(silent=True)
    if not body or 'headers' not in body or 'rows' not in body:
        return jsonify({"error": "Invalid request body; need {headers, rows}"}), 400

    headers: List[str] = body['headers']
    rows: List[Dict[str, Any]] = body['rows']

    if not headers:
        return jsonify({"error": "Headers list is empty"}), 400

    # Create a timestamped backup before overwriting.
    backup_dir = OUTPUT_DIR / "_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    backup_name = f"{full_path.stem}-{ts}.csv"
    shutil.copy2(str(full_path), str(backup_dir / backup_name))

    # Keep only the 30 most recent backups for this CSV.
    prefix = full_path.stem + "-"
    all_backups = sorted(
        [p for p in backup_dir.glob(f"{prefix}*.csv")],
        key=lambda p: p.stat().st_mtime,
    )
    for old in all_backups[:-30]:
        old.unlink(missing_ok=True)

    # Write the new CSV atomically (write to temp then rename).
    tmp_path = full_path.with_suffix('.csv.tmp')
    try:
        with tmp_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            for row in rows:
                writer.writerow({h: row.get(h, '') for h in headers})
        tmp_path.replace(full_path)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return jsonify({"error": f"Failed to write CSV: {e}"}), 500

    return jsonify({"ok": True, "rows_written": len(rows), "backup": backup_name})


@app.route('/api/csv/delete-rows/<path:csv_path>', methods=['POST'])
def api_csv_delete_rows(csv_path: str):
    """Delete rows by index.  Expects JSON body: {indices: [0, 3, 5]}."""
    full_path = _resolve_csv_path(csv_path)

    body = request.get_json(silent=True)
    if not body or 'indices' not in body:
        return jsonify({"error": "Invalid request body; need {indices}"}), 400

    indices_to_delete = set(body['indices'])

    # Read existing data.
    headers: List[str] = []
    rows: List[Dict[str, Any]] = []
    with full_path.open() as f:
        reader = csv.DictReader(f)
        headers = reader.fieldnames or []
        rows = list(reader)

    # Backup before modifying.
    backup_dir = OUTPUT_DIR / "_backups"
    backup_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    shutil.copy2(str(full_path), str(backup_dir / f"{full_path.stem}-{ts}.csv"))

    new_rows = [r for i, r in enumerate(rows) if i not in indices_to_delete]

    tmp_path = full_path.with_suffix('.csv.tmp')
    try:
        with tmp_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=headers)
            writer.writeheader()
            writer.writerows(new_rows)
        tmp_path.replace(full_path)
    except Exception as e:
        tmp_path.unlink(missing_ok=True)
        return jsonify({"error": f"Failed to write CSV: {e}"}), 500

    return jsonify({"ok": True, "deleted": len(indices_to_delete), "remaining": len(new_rows)})


if __name__ == '__main__':
    # Run on port 5000, accessible from outside container
    app.run(host='0.0.0.0', port=5000, debug=False)
