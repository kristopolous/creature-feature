#!/bin/sh
# Poll a Comfy Build release until no artifact is pending. Usage: wait_release.sh RELEASE_ID [SPEC_DIR]
set -a; . ./.env; set +a
for i in $(seq 90); do
  ../comfy/.venv/bin/comfy build release show "$1" "${2:-build}" 2>&1 | tail -1 > /tmp/claude-1000/release.json
  python3 - <<'PY' && exit 0
import json, sys, time
d = json.loads(open('/tmp/claude-1000/release.json').read(), strict=False)['data']
c = d['artifactCounts']
print(time.strftime('%H:%M'), 'ready', c['ready'], 'pending', c['pending'], 'failed', c['failed'], [a['status'] for a in d['artifacts']], flush=True)
if c['pending'] == 0:
    print(json.dumps(d['artifacts'], indent=1)[:1500])
    sys.exit(0)
sys.exit(1)
PY
  sleep 60
done
