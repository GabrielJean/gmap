# Minimal image with Playwright + Chromium dependencies
FROM python:3.11-slim-bookworm

WORKDIR /app

ENV PYTHONPATH=/app/src \
	PIP_NO_CACHE_DIR=1 \
	PYTHONDONTWRITEBYTECODE=1

# System dependencies for Chromium (slim base needs explicit runtime libs)
RUN apt-get update \
	&& DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
		ca-certificates \
		libnss3 libatk1.0-0 libatk-bridge2.0-0 libx11-xcb1 libxcomposite1 \
		libxdamage1 libxrandr2 libgbm1 libasound2 libxshmfence1 libgtk-3-0 \
		libpango-1.0-0 libcups2 libdrm2 libxkbcommon0 libxfixes3 libxext6 \
		libxrender1 libx11-6 libxcb1 libxss1 libglib2.0-0 libatspi2.0-0 \
		libcairo2 fonts-liberation tzdata \
	&& rm -rf /var/lib/apt/lists/*

# Python deps + browser binaries
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt \
	&& python -m playwright install chromium

# Application code
COPY src ./src

# Runtime configuration (decrypted by CI before docker build)
COPY .env ./

# Expose web dashboard port
EXPOSE 5000

ENTRYPOINT ["python", "-m", "gmap.app"]
