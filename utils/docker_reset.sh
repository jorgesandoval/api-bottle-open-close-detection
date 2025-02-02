#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

echo "🧹 Initiating Docker cleanup..."

# Check for any containers before attempting to stop/remove them
container_ids=$(docker ps -aq)

if [ -n "$container_ids" ]; then
  # 1️⃣ Stop all running containers
  echo "⏹️ Stopping all running containers..."
  docker stop $container_ids || { echo "❌ Failed to stop containers"; exit 1; }

  # 2️⃣ Remove all containers
  echo "🗑️ Removing all containers..."
  docker rm $container_ids || { echo "❌ Failed to remove containers"; exit 1; }
else
  echo "ℹ️ No containers found to stop or remove."
fi

# 3️⃣ Prune unused Docker objects (containers, networks, images, and volumes)
echo "✨ Pruning unused Docker objects (containers, networks, images, and volumes)..."
docker system prune -a --volumes -f || { echo "❌ Failed to prune Docker system"; exit 1; }

# 4️⃣ Verify that no containers are running
echo "🔍 Checking running containers..."
docker ps || { echo "❌ Failed to list running containers"; exit 1; }

echo "✅ Docker cleanup complete!"

