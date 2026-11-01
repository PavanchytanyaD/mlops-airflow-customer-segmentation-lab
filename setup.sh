#!/bin/bash
set -e

# Stop and remove containers, networks, and volumes from any earlier run
docker compose down -v

# Create the folders Airflow mounts into the containers
mkdir -p ./logs ./plugins ./config ./working_data

# Write the current user's UID into .env so files created by Airflow stay editable
echo "AIRFLOW_UID=$(id -u)" > .env

# Build the Airflow image with the extra packages from requirements.txt
docker compose build

# Set up the metadata database and the admin user (airflow / airflow)
docker compose up airflow-init
