# gmap

A web scraping application that uses Playwright to scrape dynamic web pages on a scheduled basis and displays the data through a web dashboard.

## Features

- Scrapes a list of URLs using Playwright with Chromium
- Runs on a quarter-hour schedule (every 15 minutes)
- Generates graphs and logs data over time
- **Web Dashboard** to view all collected data, graphs, and logs
- Containerized with Docker for easy deployment

## Web Dashboard

The application now includes a web dashboard accessible on port 5000 that displays:

- **Traffic graphs**: Visual representations of historical traffic data
- **Data tables**: Raw CSV data for each route and direction
- **Statistics**: Total records and latest update timestamps
- **Live logs**: Recent application logs
- **Auto-refresh**: Dashboard updates automatically every 2 minutes

The dashboard is available at `http://localhost:5000` (or the randomly assigned port when using `docker-compose`).

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

Alternatively, to run on a specific port (e.g., 5000), modify `docker-compose.yml`:
```yaml
ports:
  - "5000:5000"  # Map host port 5000 to container port 5000
```

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

The `.env` file is encrypted with Ansible Vault for security.