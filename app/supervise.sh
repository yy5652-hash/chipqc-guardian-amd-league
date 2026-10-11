#!/bin/bash
# Container CMD: keep the model server resident for the whole evaluation.
mkdir -p /app/output /app/corpus /app/index
while true; do
  python3 /app/server.py >> /tmp/mc3-server.log 2>&1
  echo "$(date -u +%FT%TZ) server exited with $? - restarting" >> /tmp/mc3-server.log
  sleep 2
done
