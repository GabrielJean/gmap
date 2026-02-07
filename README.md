# gmap

A web scraping application that uses Playwright to scrape dynamic web pages on a scheduled basis and displays the data through a web dashboard.

Source code lives in `src/gmap/` with:
- `src/gmap/app.py` as the container entrypoint
- `src/gmap/scraper.py` for scraping
- `src/gmap/web_app.py` for the Flask dashboard
- `src/gmap/templates/` for HTML

## Features

- Scrapes a list of URLs using Playwright with Chromium
- Runs on a quarter-hour schedule (every 15 minutes)
- Generates graphs and logs data over time
- **Web Dashboard** to view all collected data, graphs, and logs
- Containerized with Docker for easy deployment

## Web Dashboard

The application now includes a web dashboard accessible on port 5000 that displays:

- **Interactive traffic graphs**:
  - Real-time line charts showing duration over time
  - Median and 95th percentile reference lines
  - Zoom, pan, and hover capabilities for detailed analysis
- **Data tables**: Raw CSV data for each route and direction (expandable)
- **Statistics**: Total records and latest update timestamps for each route
- **Live logs**: Recent application logs for monitoring and debugging
- **Auto-refresh**: Dashboard updates automatically every 2 minutes

The dashboard reads directly from the CSV files stored in the `data_runs/` directory, ensuring all historical data is accessible.

### Features:
- Responsive design that works on desktop and mobile
- Cards for each route/direction combination
- Interactive Plotly.js charts for data visualization
- Filterable and sortable data tables
- Real-time log viewing

## Running Locally

### Prerequisites

- Docker and Docker Compose
- Ansible Vault password (for decrypting .env file)

### Setup

1. Clone the repository
2. Decrypt the .env file:
   ```bash
   ansible-vault decrypt .env
   ```
3. Run with Docker Compose:
   ```bash
   docker-compose up --build
   ```
4. Access the web dashboard:
   - Find the exposed port: `docker ps` (look for the port mapping like `0.0.0.0:xxxxx->5000/tcp`)
   - Open browser: `http://localhost:<port>` (replace `<port>` with the actual port number)
   - Or use a specific port by modifying `docker-compose.yml`:
     ```yaml
     ports:
       - "5000:5000"  # Map host port 5000 to container port 5000
     ```
   - Access at: `http://localhost:5000`

### Data Storage

All historical data is stored in CSV files in the `data_runs/` directory, which is mounted from the host at `/Apps/gmap`. The directory structure is:

```
data_runs/
├── {route-name}/
│   └── {direction}/
│       ├── {route-name}-{direction}.csv  # Historical data
│       └── {route-name}-{direction}.png  # Generated graph (optional)
├── logs.txt                               # Application logs
└── snapshots/                             # Debug snapshots
```

The web dashboard reads these CSV files to display charts and tables, ensuring compatibility with existing historical data.

## CI/CD Pipeline

This repository includes a Gitea Actions workflow that:
1. Builds the Docker image
2. Pushes to the Gitea container registry
3. Deploys to Portainer

The workflow is triggered on pushes to the `main` branch.

## Configuration

Configuration is managed through the `.env` file, which contains:
- URLs to scrape (JSON array or comma-separated list)
- Other application settings

**Important:** The `.env` file is encrypted with Ansible Vault for security. During deployment, the CI/CD pipeline automatically decrypts it before building the Docker image.

### .env Format

```bash
URLS='[{"name": "Route 1", "direction": "North", "url": "https://..."}, {"name": "Route 2", "direction": "South", "url": "https://..."}]'
```

Or simple comma-separated URLs:
```bash
URLS='https://maps.google.com/...,https://maps.google.com/...'
```

The web dashboard will display the number of configured routes on the header.