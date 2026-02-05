# Migration Guide

This document explains the migration of the gmap application from the HomeLab repository to its own repository.

## What Changed

### GabrielJean/gmap Repository (this repo)
This repository now contains:
- Application code (`gmap.py`)
- Docker configuration (`Dockerfile`, `requirements.txt`, `run.sh`)
- CI/CD pipeline (`.gitea/workflows/buildandpush.yml`)
- Environment configuration (`.env` - encrypted with Ansible Vault)
- Local development docker-compose (`docker-compose.yml`)

The CI/CD pipeline automatically:
1. Builds the Docker image on every push to `main`
2. Pushes to `gitea.docker-1.gwebs.ca/gabriel/gmap:latest`
3. Deploys to Portainer

### GabrielJean/HomeLab Repository
The HomeLab repository should now only contain:
- `docker-compose.yml` - pulls the pre-built image from the registry
- `.env` - encrypted environment configuration

**Recommended HomeLab structure:**
```
HomeLab/Docker/gmap/
├── docker-compose.yml  (uses image: gitea.docker-1.gwebs.ca/gabriel/gmap:latest)
└── .env                (encrypted with Ansible Vault)
```

**Files to remove from HomeLab:**
- `Dockerfile`
- `gmap.py`
- `requirements.txt`
- `run.sh`

These files are now maintained in this repository and automatically built/deployed via CI/CD.

## Development Workflow

### Making Changes
1. Make changes to code in the `gmap` repository
2. Commit and push to `main` branch
3. CI/CD automatically builds and deploys the new version
4. HomeLab pulls the updated image on next restart

### Local Testing
```bash
# In the gmap repository
docker-compose up --build
```

### Deploying to HomeLab
The deployment is automatic via the CI/CD pipeline. No manual intervention needed.

## Environment Variables

The `.env` file is encrypted with Ansible Vault. To edit:

```bash
# Decrypt
ansible-vault decrypt .env

# Make changes
vim .env

# Re-encrypt
ansible-vault encrypt .env
```

## Required Secrets

The following secrets must be configured in the Gitea repository settings:
- `TS_OAUTH_CLIENT_ID` - Tailscale OAuth client ID
- `TS_OAUTH_SECRET` - Tailscale OAuth secret
- `GITEA_REGISTRY_USER` - Gitea registry username
- `GITEA_REGISTRY_PASS` - Gitea registry password
- `ANSIBLE_VAULT_PASSWORD` - Password for decrypting .env file
- `PORTAINER_URL` - Portainer API URL
- `PORTAINER_API_KEY` - Portainer API key
- `PORTAINER_REGISTRY_AUTH_B64` - Base64 encoded registry auth

## Benefits of This Structure

1. **Separation of Concerns**: Application code is separate from deployment configuration
2. **Automated CI/CD**: Changes are automatically built and deployed
3. **Version Control**: All changes are tracked and can be rolled back
4. **Cleaner HomeLab**: HomeLab only contains deployment config, not application code
5. **Easier Collaboration**: Contributors can focus on the application without needing access to HomeLab
