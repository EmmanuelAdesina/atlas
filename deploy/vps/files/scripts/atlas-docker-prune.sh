#!/bin/bash
set -u
# keep the 2 newest reliastra:sha-* images (current + rollback), delete the rest
KEEP=2
docker images --format "{{.ID}}\t{{.CreatedAt}}" ghcr.io/reliaastra/reliastra | sort -t$'\t' -k2 -r | tail -n +$((KEEP+1)) | cut -f1 | xargs -r docker rmi 2>/dev/null | true
docker image prune -f 2>/dev/null
logger -t atlas-prune "docker churn guard ran"
df -h / | tail -1 | logger -t atlas-prune
