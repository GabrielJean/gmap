#!/bin/sh
# Run web server and scraper
trap "exit 0" INT TERM

QUARTER=900 # seconds

sleep_until_next_slot() {
  now=$(date +%s)
  next=$(( (now / QUARTER + 1) * QUARTER ))
  delay=$(( next - now ))
  [ "$delay" -gt 0 ] && sleep "$delay"
}

# Start web server in background
echo "Starting web dashboard on port 5000..."
python web_app.py &
WEB_PID=$!

# Give web server time to start
sleep 5

# Align first scrape run to the next quarter-hour
sleep_until_next_slot

# Run scraper in a loop
while true; do
  python gmap.py
  status=$?
  if [ "$status" -ne 0 ]; then
    echo "gmap run failed with status $status; retrying in 60s" >&2
    sleep 60
    continue
  fi

  sleep_until_next_slot
done
