#!/bin/bash
set -u
DEST=/opt/atlas-backups
LOG=/opt/atlas/data
STAMP=$(date +%Y%m%d-%H%M)
mkdir -p "$DEST"
tar -czf "$DEST/atlas-data-$STAMP.tar.gz" -C "$LOG" . 2>/dev/null
# integrity proof: the archive must open and contain the ledger
if ! tar -tzf "$DEST/atlas-data-$STAMP.tar.gz" | grep -q loop_decisions.jsonl; then
  logger -t atlas-backup "ERROR: backup $STAMP failed integrity check"
  rm -f "$DEST/atlas-data-$STAMP.tar.gz"
  exit 1
fi
logger -t atlas-backup "backup $STAMP ok ($(du -sh "$DEST/atlas-data-$STAMP.tar.gz" | cut -f1))"
# keep the last 48; also mirror the shadow ledger each time for continuity
ls -1t "$DEST"/atlas-data-*.tar.gz 2>/dev/null | tail -n +49 | xargs -r rm -f
