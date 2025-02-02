#!/bin/bash

# Exit on error
set -e

echo "🧹 Cleaning up Docker environment..."

# Stop containers and remove volumes
echo "📥 Stopping containers and removing volumes..."
docker compose down -v || { echo "❌ Failed to stop containers"; exit 1; }

# Remove unused networks
echo "🌐 Removing unused networks..."
docker network prune -f || { echo "❌ Failed to remove networks"; exit 1; }

# Remove all images
echo "🗑️ Removing all Docker images..."
docker rmi $(docker images -q) || echo "⚠️ Some images could not be removed (they might be in use)"

# Rebuild and start services
echo "🏗️ Rebuilding and starting services..."
docker compose up --build || { echo "❌ Failed to rebuild services"; exit 1; }

echo "✅ Done!"
