#!/bin/bash
set -u
cd /home/ubuntu/synctrace || exit 1
RUN_ID=$(gh run list --repo bbvizzotto-run/synctrace --limit 1 --json databaseId -q '.[0].databaseId')
echo "RUN_ID=${RUN_ID}"
gh run list --repo bbvizzotto-run/synctrace --limit 1 --json databaseId,status,name
if [ -n "$RUN_ID" ]; then
  gh run view "$RUN_ID" --repo bbvizzotto-run/synctrace --log-failed 2>/dev/null | head -80
fi
