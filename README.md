# gmap

A web scraping application that uses Playwright to scrape dynamic web pages on a scheduled basis.

## Features

- Scrapes a list of URLs using Playwright with Chromium
- Runs on a quarter-hour schedule (every 15 minutes)
- Generates graphs and logs data over time
- Containerized with Docker for easy deployment

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