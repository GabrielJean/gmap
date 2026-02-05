# Project Summary

## Overview
This repository contains the gmap web scraping application, which uses Playwright to scrape dynamic web pages on a scheduled basis (every 15 minutes).

## Migration Status
✅ **Complete** - Successfully migrated from GabrielJean/HomeLab/Docker/gmap

## Repository Structure
```
.
├── .dockerignore              # Docker build exclusions
├── .env                       # Environment variables (encrypted with Ansible Vault)
├── .gitea/
│   └── workflows/
│       └── buildandpush.yml   # CI/CD pipeline for Gitea Actions
├── .gitignore                 # Git exclusions
├── Dockerfile                 # Container image definition
├── MIGRATION.md               # Detailed migration guide
├── README.md                  # Project documentation
├── docker-compose.homelab.yml # Template for HomeLab repository
├── docker-compose.yml         # Local development compose file
├── gmap.py                    # Main application (831 lines)
├── requirements.txt           # Python dependencies
└── run.sh                     # Container entrypoint script
```

## CI/CD Pipeline
The Gitea Actions workflow (`.gitea/workflows/buildandpush.yml`) automatically:
1. ✅ Builds Docker image on push to `main` branch
2. ✅ Decrypts `.env` file using Ansible Vault
3. ✅ Pushes image to `gitea.docker-1.gwebs.ca/gabriel/gmap:latest`
4. ✅ Deploys to Portainer with automatic container recreation

## Verification Completed
- ✅ Docker build successful (tested locally)
- ✅ YAML workflow syntax validated
- ✅ Code review completed (all issues resolved)
- ✅ Security scan completed (0 vulnerabilities)
- ✅ Documentation complete

## Next Steps for HomeLab
To complete the migration, update the HomeLab repository:

1. In `HomeLab/Docker/gmap/`:
   - Delete: `Dockerfile`, `gmap.py`, `requirements.txt`, `run.sh`
   - Keep: `.env` (no changes needed)
   - Update: `docker-compose.yml` with content from `docker-compose.homelab.yml`

2. Test the updated setup:
   ```bash
   cd HomeLab/Docker/gmap
   docker-compose pull
   docker-compose up -d
   ```

## Required Secrets (Gitea Repository)
The following secrets must be configured in the Gitea repository settings:
- `TS_OAUTH_CLIENT_ID` - Tailscale OAuth client ID
- `TS_OAUTH_SECRET` - Tailscale OAuth secret  
- `GITEA_REGISTRY_USER` - Gitea registry username
- `GITEA_REGISTRY_PASS` - Gitea registry password
- `ANSIBLE_VAULT_PASSWORD` - Password for decrypting .env
- `PORTAINER_URL` - Portainer API URL
- `PORTAINER_API_KEY` - Portainer API key
- `PORTAINER_REGISTRY_AUTH_B64` - Base64 encoded registry auth

## Benefits
1. ✅ Separation of concerns (app code vs deployment config)
2. ✅ Automated CI/CD pipeline
3. ✅ Version control for all changes
4. ✅ Cleaner HomeLab repository structure
5. ✅ Easier collaboration and maintenance
