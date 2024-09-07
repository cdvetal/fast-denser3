#!/bin/bash
set -e

# Check if we are in the right place
if [ ! -f ./Dockerfile ]; then
    echo "No Dockerfile found. Are you executing this command in the 'docker' subfolder?"
    exit 1
fi

# Move to project root
cd ../

docker build --progress=plain -f docker/Dockerfile ./  -t fast-denser3
