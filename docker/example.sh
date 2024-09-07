#!/bin/bash
set -e

# Check if we are in the right place
if [ ! -f ./Dockerfile ]; then
    echo "No Dockerfile found. Are you executing this command in the 'docker' subfolder?"
    exit 1
fi

# Move to project root
cd ../

LD_PRELOAD="LD_PRELOAD=/usr/local/lib/python3.8/dist-packages/scikit_learn.libs/libgomp-d22c30c5.so.1.0.0"
docker run -it -v$PWD:/code denser3 /bin/bash -c "cd /code/f-denser && $LD_PRELOAD DEBUG=True python3 -m fast_denser.engine -d mnist -c /code/example/config.json -r 1 -g /code/example/cnn.grammar"
