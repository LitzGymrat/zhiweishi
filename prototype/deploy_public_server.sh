#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [[ ! -f .env ]]; then
  if [[ -f server.env.example ]]; then
    echo "[deploy] missing .env. For server deployment, run: cp server.env.example .env"
  else
    echo "[deploy] missing .env. Copy .env.example to .env and fill in your API keys first."
  fi
  exit 1
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "[deploy] docker is not installed. Please install Docker first."
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "[deploy] docker compose is not available. Please install the Docker Compose plugin."
  exit 1
fi

if ! grep -q '^access_password=' .env; then
  echo "[deploy] warning: access_password is not set in .env. Public deployment should set a password."
elif grep -Eq '^access_password=$' .env; then
  echo "[deploy] warning: access_password is empty. Public deployment should set a password."
fi

echo "[deploy] starting public server in read-only mode..."
docker compose -f compose.public.yml up -d --build

echo ""
echo "[deploy] done. Open: http://<your-server-ip>:8506"
echo "[deploy] if you changed PUBLIC_APP_MODE, check compose.public.yml and your env settings."