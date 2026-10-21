# Deploying ATLAS on a VPS

ATLAS drives a real browser (the Kimi WebBridge extension) to read SportyBet's
live board, prices it, and places bets. The browser is the fragile part, so
the deployment is: a browser on the box with the extension installed and
logged into SportyBet, the WebBridge daemon, and the loop.

## 1. The box

- A VPS with a desktop environment (the browser needs one). Ubuntu 22.04
  with `ubuntu-desktop` or a headless Chrome with the extension loaded.
- Chrome (or Chromium) with the **Kimi WebBridge** extension installed and
  the SportyBet session **logged in**. The session expires: the loop detects
  an expired session, reports it, and stops placing — it does not recover on
  its own, because only a human can log in.

## 2. Install

```bash
sudo useradd -m -s /bin/bash atlas
sudo apt-get install -y python3-venv google-chrome-stable
sudo -u atlas -H bash

git clone <repo> /opt/atlas
cd /opt/atlas
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt
cp .env.example .env
$EDITOR .env          # ATLAS_WEBHOOK_URL, ATLAS_HOST=0.0.0.0, ...
```

## 3. The browser and the daemon

Start the browser with the extension, log into SportyBet, then start the
daemon:

```bash
~/.kimi-webbridge/bin/kimi-webbridge daemon
~/.kimi-webbridge/bin/kimi-webbridge status
```

The daemon must report `extension_connected: true`.

Two things that will otherwise cost you an afternoon:

- **The extension only drives the FOREGROUND tab.** Every command for a
  background tab is sent and never executed, and the daemon answers 502.
  Keep the browser on the SportyBet tab, or run it headful and unfocused
  but with the tab active. If ATLAS reports "the browser session is not
  responding", click the SportyBet tab before anything else.
- **Keep the extension and the daemon on the same version.** The extension
  auto-updates; the daemon does not. When they drift, the daemon sends
  commands the extension ignores, with the same 502 and no other symptom.
  `kimi-webbridge upgrade` (or install the binary by hand and check the
  sha256 against the manifest) after any extension update.

## 4. Services

```bash
sudo cp deploy/*.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now kimi-webbridge.service
sudo systemctl enable --now atlas-loop.service
sudo systemctl enable --now atlas-dashboard.service
```

The loop service runs **observe-only**: it plans and prices but does not
place. That is deliberate.

## 5. Prove it out, in this order

```bash
cd /opt/atlas

# 1. The notification channel works.
.venv/bin/python -m atlas.main posture "balanced"
#    -> a notification should arrive at ATLAS_WEBHOOK_URL

# 2. Capture, price, plan, decide — with NO placement.
.venv/bin/python -m atlas.main run --cycles 5 --cycle-seconds 30
#    -> read data/decisions/loop_decisions.jsonl: 'recommend' entries are
#       the bets the system WOULD have placed

# 3. Only now arm execution, with the daily stop in place.
.venv/bin/python -m atlas.main run --submit --cycles 1 --cycle-seconds 30
```

Before step 3, make sure:
- the account is funded and the balance is above the NGN 100 minimum;
- `--bankroll` is set to the REAL balance (the default 100,000 is a
  placeholder — the stake is sized as a fraction of the bankroll, and with
  the 5pp model-market gap guard a small bankroll can never justify the
  NGN 100 minimum);
- the daily stop is at -NGN 500 (the default).

## 6. Going live

```bash
# Edit the service to arm execution.
sudo sed -i 's/--cycle-seconds 30/--submit --cycle-seconds 30/' /etc/systemd/system/atlas-loop.service
sudo systemctl restart atlas-loop
```

Then watch it:

```bash
journalctl -u atlas-loop -f
tail -f /var/log/atlas/loop.log
```

## Operating notes

- **The live board is never navigated directly.** `navigate()` to
  `/sport/live` times out (the page never fires `load`) and unregisters the
  tab, which 502s every later call. The loop reaches the board through a
  light page and a client-side navigation.
- **The extension wedges** after heavy use and the daemon returns 502. The
  capture recovers by loading a light page. If it does not recover, restart
  the daemon: `sudo systemctl restart kimi-webbridge.service`.
- **The session expires.** The loop reports it through the webhook and stops
  placing. Log into SportyBet again in the browser.
- **The dashboard** is on `ATLAS_HOST:ATLAS_PORT`. It can set the posture
  and read the state. It cannot place a bet.
- **The ledger** (`data/ops/ledger.jsonl`) and the decision log
  (`data/decisions/loop_decisions.jsonl`) are the audit trail. Back them up.
