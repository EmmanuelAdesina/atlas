#!/opt/atlas/.venv/bin/python
"""atlas-cdp-deep-probe — the CDP liveness probe that CANNOT lie.

GET /json/version can answer 200 while the tab-level DevTools channel is
starved (2026-11-02 15:22-15:26 wedge: HTTP green, every evaluate timed
out for the whole readiness window). The loop's capture and the settle
scrape both live on the websocket layer, so THAT is the plane this probe
tests: one real Runtime.evaluate roundtrip on an existing page target.

exit 0  -> HTTP /json/version 200 AND a WS evaluate returned 1+1 == 2
exit 42 -> any layer dark (http refusal, no page target, ws timeout,
           wrong answer). Callers MUST treat 42 as "recycle or decline".
"""
import json
import sys
import time
import urllib.request

HOST = "http://127.0.0.1:9222"


def main() -> int:
    try:
        with urllib.request.urlopen(f"{HOST}/json/version", timeout=5) as r:
            if r.getcode() != 200:
                print(f"http /json/version -> {r.getcode()}", file=sys.stderr)
                return 42
        pages = [
            t
            for t in json.load(urllib.request.urlopen(f"{HOST}/json/list", timeout=5))
            if t.get("type") == "page" and t.get("webSocketDebuggerUrl")
        ]
        if not pages:
            print("no page targets on /json/list", file=sys.stderr)
            return 42
        ws_url = pages[0]["webSocketDebuggerUrl"]
    except Exception as exc:  # noqa: BLE001 — any transport failure = wedged
        print(f"transport probe failed: {str(exc)[:120]}", file=sys.stderr)
        return 42

    try:
        import websocket  # venv-only dependency, same as the scrape
    except ImportError as exc:
        print(f"websocket-client missing: {exc}", file=sys.stderr)
        return 42

    try:
        ws = websocket.create_connection(ws_url, timeout=8)
    except Exception as exc:  # noqa: BLE001
        print(f"ws connect failed: {str(exc)[:120]}", file=sys.stderr)
        return 42

    try:
        ws.send(
            json.dumps(
                {
                    "id": 1,
                    "method": "Runtime.evaluate",
                    "params": {"expression": "1+1", "returnByValue": True},
                }
            )
        )
        deadline = time.monotonic() + 8.0
        while time.monotonic() < deadline:
            ws.settimeout(max(1.0, deadline - time.monotonic()))
            msg = json.loads(ws.recv())
            if msg.get("id") == 1:
                val = (msg.get("result") or {}).get("result", {}).get("value")
                return 0 if val == 2 else 42
    except Exception as exc:  # noqa: BLE001 — timeout/closed/parse = wedged
        print(f"ws evaluate failed: {str(exc)[:120]}", file=sys.stderr)
        return 42
    finally:
        try:
            ws.close()
        except Exception:
            pass

    print("no evaluate response inside deadline", file=sys.stderr)
    return 42


if __name__ == "__main__":
    sys.exit(main())
