"""Web dashboard for viewing gmap traffic data."""

from __future__ import annotations

import json
import csv
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Optional

from flask import Flask, render_template, jsonify, send_file, abort
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
        # Load URLs from .env (same method used by gmap.py)
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


if __name__ == '__main__':
    # Run on port 5000, accessible from outside container
    app.run(host='0.0.0.0', port=5000, debug=False)
