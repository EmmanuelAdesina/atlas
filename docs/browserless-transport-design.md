# Browserless Transport Architecture — Design

**Status:** DESIGN ONLY. No code in this document is implemented. No execution,
submission, campaign, gate, stake, or operator-authorization behavior is changed
by this design.
**Date:** 2026-11-03
**Scope:** External interaction layer of Atlas (SportyBet `sportybet.com/ng`).
The pricing kernel, calibration, decision gates, ledger, campaign wall and
boss/posture planes are out of scope and remain untouched.

---

## 1. Existing architecture and dependency analysis

### 1.1 What Atlas is today

One process, two planes:

- **Fast (deterministic) plane** — browser-independent by construction:
  `atlas/live/math_engine.py` (Dixon-Coles joint matrix, 14+ market families),
  `atlas/live/paths.py` (path-dependent contracts), `atlas/live/estimate.py`
  (state-space, `[p_lower, p_upper]`), `atlas/live/calibration.py` (PAVA
  isotonic), `atlas/live/decisions.py` (gate chain), `atlas/live/loop.py`.
  None of these import or touch the browser.
- **Slow (control) plane** — also independent of the browser:
  `atlas/ops/ledger.py` (append-only JSONL), `atlas/ops/campaign.py`
  (pre-registered verdict + fail-closed wall), `atlas/ops/boss.py` /
  `atlas/live/boss.py`, `atlas/ops/posture.py`, `atlas/live/control.py`
  (dashboard). The LLM supervisor has no pricing/execution authority.
- **I/O rim** — everything that needs Chrome lives here:
  `atlas/browser/webbridge.py` (Kimi daemon adapter, largely decommissioned),
  `atlas/browser/cdp_direct.py` / `cdp_first.py` (raw DevTools transport, now
  primary), `atlas/live/capture.py` (board/match/markets reading),
  `atlas/browser/betslip.py` (the only money-moving code),
  `atlas/browser/cashout.py`, `atlas/browser/slipclear.py`,
  `atlas/live/bet_history_scrape.py` (settlement truth),
  `atlas/live/supervisor.py` (re-login).

LOC baseline (the entire browser/transport rim): `atlas/browser/*.py` ≈ 2,266
LOC, `atlas/live/capture.py` 1,619 LOC, `atlas/live/bet_history_scrape.py`
801 LOC → **≈ 4,700 LOC of browser-coupled code**, vs. `math_engine.py` 1,659
LOC and `decisions.py`-class gate logic of comparable size. The riskiest,
most incident-dense part of the system is the rim, not the engine.

### 1.2 Cycle lifecycle and Chrome dependency map

Normal `atlas.main run --submit` cycle (`atlas/live/runner.py`):

```
capture() ──────────────────► CHROME REQUIRED
  readiness gate → open_match → read markets (JS injection) → scoreboard/stats
session_expired() probe ────► CHROME REQUIRED (header `NGN <balance> | Deposit`)
board_candidates() ─────────► CHROME REQUIRED (multi-view row scan)
price / gates / kelly ──────► no browser
enqueue_place → place_bet() ► CHROME REQUIRED (real two-step click flow;
                              money moves only here)
read_balance_ngn() ─────────► CHROME REQUIRED
settle_finished() ──────────► CHROME REQUIRED (bet_history_scrape)
handle_cashout_offers() ────► CHROME REQUIRED (CashoutDriver reads/executes)
ledger / campaign / posture ► no browser
```

What Chrome is doing per cycle is ~100% *sensing* and ~0% *deciding*. Chrome
is also doing things it does not need to do at all: authentication state is a
cookie jar, and every payload it shuttles is JSON over HTTP. This is the core
inefficiency the redesign removes.

### 1.3 Current failure evidence (operational, not theoretical)

- `data/ops/heartbeat.json` (2026-11-03T14:30Z, `cycle: 218`):
  `last_error: "board scan failed — browser automation unreachable … HTTP 502"`,
  `stage: "cycle 218: capture"`, `priced_selections: 0`, `submission_enabled:
  true` but `ready: false`. Balance NGN 111.96. i.e. the engine is idle because
  the *sensor* is down, not because of any decision.
- `deploy/vps/units/atlas-browser.service`: `MemoryMax=900M` (OOM magnet),
  `MemoryHigh=650M` had to be toggled repeatedly; on-box incidents show
  synchronous reclaim wedging Chrome to 0 progress (`atlas-browser-notes.md`);
  docs verdict: "needs 4 GB RAM or a separate box".
- `docs/bugs_and_issues.md`: B27 (`navigate()` to the board always times out —
  websocket page never fires `load`), B13 (network capture filter
  case-sensitivity made the documented network path silently unused), B35
  (login-button race; transient 502s the dominant failure class).
- Settle path works only by driving a *second* browser session
  (`ATLAS_BET_HISTORY_TRANSPORT=cdp`, separate tab, `atlas-settle` timer).

### 1.4 Verification corrections (claims vs. evidence)

Every important claim was checked against artifacts and source:

| Claim | Verdict | Evidence |
|---|---|---|
| "realbetlist was reproduced" | **CONFIRMED — but from PAGE CONTEXT ONLY, never from an external client.** It used same-origin sync XHR with `withCredentials` from inside the loaded SPA. | `atlas/live/bet_history_scrape.py` (`_build_page_js`, `x.open('GET', API_PATH…)`), response body archived at `data/ops/betsdetail_temp.json` (`bizCode:10000`, `totalNum:159`) and `data/ops/realbetlist_2026-10-28.json` (`totalNum:159`) |
| "The API only answers same-origin XHR" | **UNVERIFIED as stated.** What was probed (async `fetch` *through the daemon*) got the site-level error page. An external client was never tried, or tried and not recorded. | module docstring of `bet_history_scrape.py` |
| "factsCenter/Outcomes exists and carries the board" | **CONFIRMED it exists; payload/schema UNKNOWN.** Seen once as a completed POST 200 in 2026-10 net capture; request/response never recorded. | `data/ops/net1.json` (`requestId 13352.22733`) |
| "Exact request headers are known" | **PARTIALLY.** One full header set captured for the realbetlist GET (see §2). factsCenter headers unknown. | `data/ops/betsdetail_temp.json` (sanitized here) |
| "Zero SportyBet endpoints reproduced outside Chrome" | **CONFIRMED.** The only external HTTP client in the repo that targets a live service is the Polymarket collector (`atlas/collectors/polymarket.py` via `atlas/http/client.py`) — different venue, no auth. | grep for `/api/ng/`, `graphql`, `sportybet.com` (page URLs only) |
| "Engine is browser-free" | **CONFIRMED.** No import of `atlas.browser*` outside `atlas/live/runner.py` orchestration and `bet_history_scrape.py`. | import graph check |
| "536 tests, all browser-free" | **CONFIRMED** (mocked transports; `pytest tests -q` in CI). | `pyproject.toml`, `tests/`, `.github/workflows/ci.yml` |
| Prior proposal's "~10 MB resident for the new transport" | **ESTIMATE, unverified.** To be replaced by measured values (§10). | this document §10 |
| Direct order API is usable | **REJECTED FOR NOW.** `POST /api/ng/orders/order` returns 200 on refusal; never called by Atlas by policy. Design keeps it out of scope (§9). | `docs/bugs_and_issues.md` |

### 1.5 Dependency map (browser)

```
                        ┌──────────────┐
                        │  Chrome/CDP  │   Xvfb, MemoryMax=900M
                        └───┬────┬─────┘   repo: /chrome-profile (gitignored)
              reads (all)   │    │ money (rare)
   ┌───────────────┬────────┴──┐ ▼
   │ capture.py    │ betslip.py│  cashout.py (execute)
   │ board rows,   │ two-step  │
   │ markets,      │ click     │
   │ balance,      │ flow      │
   │ session probe │
   └───────┬───────┘           │
           ▼                   ▼
      LiveRunner ───► decisions/gates ───► ledger ───► campaign wall
           ▲                                        (browser-free)
   ┌───────┴────────────────────────────┐
   │ bet_history_scrape.py (CDP/daemon) │──► settle_state.json ──► settle_history
   └─────────────────────────────────────┘
```

Removing sensors from the browser leaves exactly one irreducible browser use:
**actuation** (§9). Everything else is JSON.

---

## 2. Confirmed endpoint inventory (evidence-backed)

All hosts: `https://www.sportybet.com`. Session = cookies from the logged-in
profile; **no Authorization header observed anywhere**; no API keys exist.

Common envelope (observed on the one fully-captured response):
`{"bizCode": 10000, "isAvailable": true, "message": "Success", "data": {...}}`
— `bizCode != 10000` / `isAvailable == false` must be treated as an error
state, never as data.

| # | Endpoint | Method | Evidence status | Observed |
|---|---|---|---|---|
| 1 | `/api/ng/factsCenter/orderedSportList?_t=…` | GET | **Observed 200** (net1.json, netlist_temp.json) | menu/sport ordering; semantics of use unknown |
| 2 | `/api/ng/factsCenter/sportList?option=1&productId=1&_t=…` | GET | **Observed 200** (both captures) | params captured; body unknown |
| 3 | `/api/ng/factsCenter/event?_t=…` | GET | **Observed 200** (net1.json) | likely fixture/event detail; params+body unknown |
| 4 | `/api/ng/factsCenter/Outcomes` | POST | **Observed 200** (net1.json); **payload & body never captured** | presumed: the odds board (primary target) |
| 5 | `/api/ng/factsCenter/flexiblebet/v2/getOddsKey?_t=…` | GET | **Observed 200** (both captures) | role unknown (odds-key for bet params?) |
| 6 | `/api/ng/factsCenter/banned/events?_t=…` | GET | **Observed 200** (both captures) | banned/suspended fixtures — maps to existing "suspended never priced" rule |
| 7 | `/api/ng/orders/order/v2/realbetlist?isSettled=10&pageSize=N&pageNo=P[&_t=…]` | GET | **CONFIRMED BODY** (2 archives) | settled tickets; `totalNum=159/264`; winningStatus 20=won/30=lost/40=void; `createTime` epoch ms |
| 8 | `/api/ng/orders/config/cutbet` | POST | Observed 200 (netlist only) | presumed cashout config; untested |
| 9 | `/api/ng/realSportsGame/cashAbleBets/count?_t=…` | GET | Observed 200 (netlist only) | presumed cashable count badge |
| 10 | `/api/ng/pocket/v1/finAccs/finAcc/userBal/NGN?_t=…` | GET | Observed 200 (netlist only) | presumed balance |
| 11 | `/api/ng/pocket/v1/wallet/assetsInfo?_t=…` | GET | Observed 200 (netlist only) | wallet assets |
| 12 | `/api/ng/patron/account/info?_t=…` | GET | Observed 200 (netlist only) | account/session probe candidate |
| 13 | `POST /api/ng/orders/order` | POST | Exists; **200-on-refusal**; Atlas never calls it; **out of scope** | execution paths only |
| 14 | `POST /api/ng/orders/config/cutbet` etc. (cashout execute) | POST | Exist; untested; **out of scope** | cashout execution boundary (§9) |

### 2.1 Confirmed request header set (realbetlist GET, sanitized)

```
user-agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36
            (KHTML, like Gecko) Chrome/155.0.0.0 Safari/537.36
referer:    https://www.sportybet.com/ng/my_accounts/bet_history/sport_bets?isSettled=10
accept:     */*                      (capture shows `*/*`; page sets `application/json, text/plain, */*` on the XHR)
accept-language: en
platform:   web
clientid:   web
operid:     2
sec-ch-ua:  "Google Chrome";v="155", "Chromium";v="155", "Not(A:Brand";v="24"
sec-ch-ua-mobile: ?0
sec-ch-ua-platform: "Windows"
content-type: application/x-www-form-urlencoded; charset=UTF-8   (on a GET; replicate as observed)
cookie:     <redacted — sb_fs_id, device-id, sb_country=ng, locale=en, usrId, …>
```

No signing, no CSRF token observed on this GET. Whether factsCenter POSTs are
signed/keyed is **unknown** (§3).

### 2.2 Confirmed response schema (realbetlist, from archives)

```
{"bizCode": 10000, "isAvailable": true, "message": "Success",
 "data": {"pageSize": int, "pageNo": int, "totalNum": int,
          "entityList": [ {orderId, shortId, orderType, totalStake: str,
             selectionSize, totalWinnings: str, winningStatus: int(20/30/40),
             createTime: epoch_ms, currency, combinationSize, shareCode,
             selections: [ {eventId, marketId, outcomeId, matchStatus,
                eventStatus, odds: str, marketDesc, outcomeDesc, home, away,
                categoryName, tournamentName, startTime, ...} ], ...} ]}}
```

Pagination: `pageSize` ≤ 50 observed (scraper uses 50), `pageNo` 1-based,
`totalNum` authoritative for completeness checks (`fetched == totalNum`).

---

## 3. Unknown endpoint and reconnaissance requirements

Method for resolving these, in order of preference (no Kimi dependency):

1. **Read-only capture through the existing CDP transport**
   (`Network.getResponseBody` / fetch-domain pause) during one normal manual
   session on the live site: open the live board, open one match panel, open
   Bet History. Capture **raw** request/response pairs.
2. If the daemon/CDP cannot surface a body, a **controlled, temporary Chrome
   session used strictly for recon** (same ceremony as the existing
   `Start-Chrome` scripts), torn down afterwards. Never in runtime.
3. Nothing else — no guessed schemas, no synthesized fixtures.

Sanitized fixtures (cookies/headers stripped, PII removed) get archived to
`tests/fixtures/api/` **before** any model mapping, mirroring existing
snapshot discipline.

**Required recon outputs (Phase A):**

| # | Unknown | Why it blocks | Verification method |
|---|---|---|---|
| U1 | `Outcomes` POST: exact body (JSON? form? fixtureIds list? market bitmap?), required headers, response schema | The entire board read path | CDP capture of one board load + one match open |
| U2 | Whether factsCenter calls require the `sec-ch-ua*` family or only `platform/clientid/operid` | Transport header policy | replay experiment (§11 Phase A, off-peak) |
| U3 | `getOddsKey` semantics (needed for placement-shaped reads? for cashout?) | If it's an order-time dependency, keep it out of reads | capture + interpretation with the captured flow |
| U4 | `banned/events` shape and refresh behavior | Suspended-market correctness | capture once per session state |
| U5 | Balance endpoint chosen as session-health probe: `userBal/NGN` vs `assetsInfo` vs `patron/account/info` — which is cheapest/most stable | Liveness probe design (§7) | time 5 calls each; pick one |
| U6 | Cashout read path: does `cashAbleBets/count` + an undocumented `cashAbleBets` list return the same fields as the DOM parse (stake/offered/potential win/legs)? | CashoutSource design | capture during an open position (may not exist today) |
| U7 | `bizCode` taxonomy: success vs. auth-expired vs. rate-limit; HTTP status vs. bizCode error mapping | Fail-closed correctness | observe on a deliberately stale cookie |
| U8 | Session cookie lifetime & idle timeout; behavior on expiry (401? bizCode? login HTML page?) | Session lifecycle (§7) | observe over time; test with an expired cookie |
| U9 | Response freshness on live odds (server TTL, `Cache-Control`, `_t` buster need) and empirical rate tolerance | Staleness/poll design | repeated captures ~1 Hz for 10 min |
| U10 | Cloudflare/TLS fingerprinting behavior for non-browser clients (JA3/h2) | httpx vs curl_cffi decision (§13 R1) | replay with httpx; on block, replay with curl_cffi impersonation |
| U11 | `sportList`/`event`/`orderedSportList` exact roles (board bootstrap vs. navigation) | Minimal sufficient read set | capture a fresh page load end-to-end |
| U12 | Error body shape for 4xx/5xx (HTML vs JSON envelope) | Malformed-response handling | capture on forced failure (bad path) |

**Explicitly not part of recon:** place-order semantics, accept-changes, betslip
API internals. Those belong to the money-path decision (§9), not to sensing.

---

## 4. Proposed target architecture

```
                 ┌───────────────────────────────┐
                 │ SessionProvider (cookie vault)│  §7
                 │  chmod-600 file, seed once,   │
                 │  probe validate, fail closed  │
                 └───────────────┬───────────────┘
                                 │ cookies + header profile
                 ┌───────────────┴───────────────┐
                 │ TransportClient (httpx)       │  §8
                 │  explicit timeouts, bounded   │
                 │  pool, per-endpoint policy,   │
                 │  pre/unknown/accepted taxonomy│
                 └───────────────┬───────────────┘
          ┌───────────┬─────────┼───────────┬───────────┐
          ▼           ▼         ▼           ▼           ▼
     BoardSource BalanceSrc SettleSource CashoutSrc  (Executor port,
     §6.1       §6.2       §6.3          §6.4        §9 — unchanged)
          │           │          │           │
          └─────┬─────┴──────────┴───────────┘
                ▼
   Archive raw response FIRST (data/snapshots/…, existing discipline)
                ▼
   Pydantic schema validation (fail → transport error, fail closed)
                ▼
   Normalizers → the SAME internal dataclasses the engine consumes today
                ▼
   UNCHANGED: math_engine → estimate → calibration → decisions → gates
                → runner → ledger → campaign
```

Design rules:

1. **Transport is a dumb pipe.** No pricing, no gating, no verdicts. It returns
   typed observations or typed errors — never guesses.
2. **Archive-then-map.** Raw bytes are persisted before parsing, exactly like
   `atlas/collectors/` and `snapshots.py` do. Schema drift becomes a detectable,
   replayable event, not a silent corruption.
3. **Normalize to the existing internal shape** (`LiveMatchState`,
   `CaptureResult`, `markets` entries) so downstream (`extract`, `mappers`,
   `math_engine`) is untouched. The API becomes a *new source for the same
   observations*, not a parallel engine.
4. **Fail closed everywhere.** Unreadable transport ⇒ `ready=false` ⇒ campaign
   wall shut. "0 rows" is never an empty board; it is an error.
5. **No permanent browser fallback in runtime.** The DOM/CDP adapter remains in
   the repo as (a) the shadow-parity harness for migration and (b) a
   break-glass diagnostic. It is not wired into the operational loop after
   Phase E sign-off. (Challenge to the directive: deleting it entirely would
   destroy the only ground-truth parity check we have; retention is for tests,
   not runtime.)
6. **No new services.** No Celery/Redis/sidecars. One process, sync HTTP,
   existing data files.

---

## 5. Module and package structure

```
atlas/transport/
  __init__.py
  errors.py          # TransportError taxonomy (§8): PreSendError,
                     # OutcomeUnknownError, RejectedError, SessionExpiredError,
                     # SchemaDriftError, StaleDataError
  client.py          # TransportClient (sync httpx), header profile, retry policy
  session.py         # SessionProvider: vault load/save/validate/redact (§7)
  snapshots.py       # archive_raw() — thin wrapper over existing snapshot infra
  schemas.py         # Pydantic models: RealbetlistPage, OutcomesBoard (Phase A),
                     # BalanceInfo, CashableBets, Envelope(bizCode/message/data)
  sources/
    __init__.py
    board.py         # BoardSource      — live board + one match panel
    balance.py       # BalanceSource    — balance + session-health probe
    settle.py        # SettleSource     — realbetlist pages → settle_state.json
    cashout.py       # CashoutSource    — offers + availability (read-only)
  normalize.py       # API payloads → LiveMatchState / CaptureResult / entries
tests/fixtures/api/  # sanitized raw fixtures (Phase A deliverable)
tests/test_transport_*.py
```

`atlas/browser/*` is unchanged except one thin usage note: post-migration the
runner imports nothing from it unless actuation option A (§9) is chosen.
`atlas/live/capture.py` is *not* rewritten; `ApiCapture` implements the same
protocol surface the runner consumes (`capture()`, `board_candidates()`,
`read_balance_ngn()`, `session_expired()`, …) and both are selectable via
`ATLAS_TRANSPORT_READ=api|dom` (default `dom` until Phase D gates pass).

Explicitly rejected: reusing `AtlasHTTPClient` (`atlas/http/client.py`) — it is
async, retries every timeout/connect error indiscriminately, is Polymarket-
shaped, and has no session-vault concept. Reusing it would couple a money-
adjacent path to an exploratory collector policy. No new dependencies beyond
`httpx` + `pydantic` (+ `curl_cffi` strictly under the §13 R1 gate).

---

## 6. Interface and data-model designs

### 6.1 BoardSource

```python
class BoardSource(Protocol):
    def board_candidates(self) -> list[BoardCandidate]           # live fixtures
    def open_match(self, fixture_id: str) -> MatchHandle         # client-side state
    def read_scoreboard(self) -> ScoreState                       # minute, score, period
    def read_markets(self) -> list[MarketEntry]                   # odds grid
    def read_capture_bundle(self) -> CaptureResult                # parity w/ DOM path
```

Models (pydantic, `extra="ignore"`, `bizCode` enforced):

- `Envelope[T]` — `bizCode:int, isAvailable:bool|None, message:str, data:T`;
  `bizCode != 10000` → `RejectedError(message)`.
- `OutcomesBoard` — shape TBD by U1; stored raw regardless.
- `MarketEntry` — reuse existing dict shape (`marketId/outcomeId/odds/status`)
  after normalization; pricing layer untouched.

### 6.2 BalanceSource

```python
class BalanceSource(Protocol):
    def read_balance_ngn(self) -> float          # raises SessionExpiredError
    def probe_session(self) -> SessionHealth      # cheap; for heartbeat
```

### 6.3 SettleSource

```python
class SettleSource(Protocol):
    def fetch_settled(self, max_pages: int = 10) -> list[SettledTicket]
    def write_settle_state(self, out: Path) -> Path   # same file format as today
```

Reuses `parse_ticket()`, `WINNING_STATUS`, `_build_state()` semantics from
`bet_history_scrape.py` — those are browser-free already and are lifted, not
rewritten. `settle_history.py` consumes `settle_state.json` unchanged.

### 6.4 CashoutSource (read-only)

```python
class CashoutSource(Protocol):
    def list_offers(self) -> list[CashoutOffer]   # same dataclass as CashoutDriver
```

Execution (`CashoutDriver.execute`) stays out of the transport layer.

### 6.5 SessionProvider

```python
class SessionProvider(Protocol):
    def load(self) -> Session            # vault; raises SessionExpiredError on absence
    def validate(self) -> SessionHealth  # 1 cheap authenticated probe (U5)
    def redacted_headers(self) -> dict   # for logs: cookie → "sb_fs_id=…[redacted]"
```

### 6.6 TransportClient

```python
class TransportClient(Protocol):
    def get(self, path, *, params=None, expect="json", retryable=True) -> Envelope-ish
    def post(self, path, *, body=None, retryable=False) -> Envelope-ish
```

Bounded pool (`max_connections=4`), explicit `connect=5s / read=10s / write=5s /
pool=5s` timeouts, **no retry unless `retryable`**; GETs idempotent ⇒ ≤2
retries with jittered backoff on pre-send/5xx only; POSTs never auto-retried
(§8).

---

## 7. Session lifecycle design

**Current state:** session = cookies in `chrome-profile/` (174 MB, gitignored);
probe = header scrape of `NGN <bal> | Deposit`; re-login = LLM-supervised DOM
fill with phone/password from env; all failure-ambiguous.

**Target:**

1. **Vault.** `data/ops/session.json`, mode `0600`, directory `0700`, contents
   `{cookies: {…}, seeded_at, platform: "web", clientid: "web", operid: 2,
   user_agent: <exact captured UA>}`. Never in git (`.gitignore` already covers
   `data/`), never in logs/telemetry (only `redacted_headers()` is loggable).
2. **Seeding — operator-driven, one time, minutes:**
   - Option A (preferred): during a manual logged-in Chrome session, one CDP
     `Network.getAllCookies` pull writes the vault. No password ever handled.
   - Option B: operator exports cookies via devtools manually + `atlas.session
     import` CLI.
   - Option C (fallback): re-seed from a fresh login whenever 401/expired.
   - The vault is **seeded, not minted**: no programmatic login exists or is
     planned (respects "do not build browser-based automatic login").
3. **Validation.** Cheapest reliable authenticated probe (U5), e.g.
   `userBal/NGN` or `patron/account/info`, ≤ 1 call per heartbeat. Two
   consecutive auth failures ⇒ `SessionExpiredError` ⇒ transport offline ⇒
   campaign stays fail-closed ⇒ operator notified via existing `notify.py`.
   **Fail explicitly; never report a healthy session that wasn't verified.**
4. **Expiry & re-auth.** Expiry ⇒ all sources raise `SessionExpiredError`;
   runner treats it exactly like today's 502 wall-shut state. Recovery =
   operator re-seeds (Option A/B). No automatic retry storms, no credential
   storage beyond the vault.
5. **Redaction & handling rules.** Vault path never printed; cookie values
   replaced by name+`[redacted]` in any log/snapshot/error; snapshot archiver
   strips `cookie`/`set-cookie` headers; CI fixture check greps for `sb_fs_id`
   patterns to prevent fixture leakage.
6. **Rotation.** Vault stores `seeded_at`; age > 25 days (typical book session
   appetite — to be measured, U8) ⇒ warning heartbeat field; age > 30 days ⇒
   validation forced before any trust.
7. **PII note.** The captured recon artifacts contain live session cookies and
   a phone number (`data/ops/betsdetail_temp.json`). They are gitignored, but
   the design **requires** sanitizing them into `tests/fixtures/api/` (cookies
   removed) and recommends the operator rotate the SportyBet password/session
   after this recon is encoded, since the cookie was handled in plaintext
   tooling. Flagged here as a security action item, not done silently.

---

## 8. Failure-state and retry matrix

### 8.1 Error taxonomy (transport-level; distinct from HTTP status)

| Class | Meaning | Examples |
|---|---|---|
| `PRE_SEND` | request demonstrably never left the client | DNS failure, connect refused, TLS mismatch, validation error |
| `OUTCOME_UNKNOWN` | sent, outcome ambiguous | read timeout after request sent, connection reset mid-body, 5xx after server accepted |
| `ACCEPTED` | server affirmatively accepted (2xx + valid envelope) | board payload, ticket page |
| `REJECTED` | server affirmatively refused | 4xx, `bizCode != 10000`, site-level error page HTML |
| `VERIFIED` | outcome independently confirmed by a second read | order seen in realbetlist; **money path only** |

A 200 status is **never** acceptance proof for orders; for reads it is
acceptance only *after* schema validation.

### 8.2 Retry policy

| Condition | Retries | Behavior |
|---|---|---|
| DNS/connect failure (PRE_SEND) | ≤2, backoff+jitter | then `TransportUnavailable` → fail closed |
| Read timeout on GET | 1 | response discarded; never reuse bytes |
| Read timeout on POST | **0** | `OUTCOME_UNKNOWN`; never blind-resend |
| HTTP 401/403 or auth `bizCode` | 0 | `SessionExpiredError`; vault flagged; notify; fail closed |
| HTTP 429 | 0–1 (honor `Retry-After`, cap 5 s) | beyond that: fail closed this cycle |
| HTTP 5xx on GET | ≤2, backoff | then fail closed |
| HTTP 5xx on POST | 0 | `OUTCOME_UNKNOWN` |
| Malformed JSON / schema drift | 0 | persist raw to `data/quarantine/`, raise `SchemaDriftError`; fail closed |
| Empty payload (`totalNum=0`, 0 markets) | 0 | **not** an empty board — `UnreadableBoard` |
| Partial market coverage (<95% priced coverage of quoted cells, mirroring B29 harness) | n/a | `PartialCoverage` warning; pricing proceeds on covered families only, markets flagged — never force-priced |
| Stale odds (payload age > TTL, default 10 s live / 60 s prematch) | 0 | `StaleDataError`; cycle abstains |
| Session probe flaky (1 of 2 fails) | 1 re-probe | then expired-session path |
| Sequencing: inflight placement | n/a | unchanged runner queue semantics |

Retries apply **only** to idempotent GETs and only pre-acceptance. No financial
or order-adjacent call is ever retried implicitly. Placement, confirmation and
cashout-execute semantics are untouched (§9).

### 8.3 Propagation into the engine

Transport errors become typed reasons (`unreadable_board`, `session_expired`,
`stale_odds`, `partial_coverage`) attached to `CaptureResult.ready=False` and
heartbeat. Decisions layer already abstains on `ready=False`; no gate logic
changes. `0 rows ≠ empty board` remains an invariant, now enforced by schema
and envelope validation rather than by renderer heuristics.

---

## 9. Security and execution-boundary analysis

**Non-negotiables preserved:** execution stays opt-in (`--submit` +
`allow_submit=True` + campaign wall + `operator_arm`), stake caps, NGN-10 unit,
one-way posture ratchet, append-only ledger, no LLM authority over money.

**Sensors vs. actuator.** This design moves *sensing* to HTTP. It deliberately
does **not** touch the actuator:

| Option | Placement path | Verdict in this design |
|---|---|---|
| A — keep real clicks | Chrome used only for the ~seconds around an actual bet; zero reads through it | **Recommended default.** Keeps the battle-tested two-step click + silent-slip verification; browser can be a minimal, restartable `--headless=new`-free Xvfb instance or a separate small box; not in the read path |
| B — direct `POST orders/order` | Pure HTTP placement | **Out of scope.** 200-on-refusal semantics, no idempotency key, ambiguous timeouts; requires (i) separate engineering decision, (ii) explicit operator approval, (iii) idempotency design: deterministic client-ref, pre-flight slip check, post-submit verification loop against realbetlist (VERIFIED state), duplicate prevention via ledger short-circuit, campaign wall unchanged |
| C — current CDP/DOM for everything | status quo | rejected by this redesign |

Cashout **execution** likewise unchanged (still behind `--submit`; read side
moves to CashoutSource). Order reconciliation (Option B would need it): the
realbetlist endpoint is the independent verifier; design sketch recorded here
so the separate decision starts from evidence, not from scratch — but **no
money-path code is proposed, and nothing is enabled by this document.**

**Threat model delta.** Removing Chrome removes a 174 MB profile, its cookie DB
at rest in a gitignored dir, a websocket-driven attack surface, and a
privileged logged-in renderer — and replaces it with a 600-byte vault.
Residual risks: vault theft (mitigated: perms, redaction, rotation U8), API
misuse detection by the book (mitigated: poll cadence ≈ the book's own SPA
cadence, no order calls), legal/ToS exposure on any scripted use (operator's
call; unchanged from today).

**PII/handles in repo artifacts:** flagged in §7 item 7 — quarantine + sanitize
during Phase A; advise session rotation.

---

## 10. Resource consumption — baseline and measurement plan

### 10.1 Measured baseline (this repo / this box)

| Metric | Value | Source |
|---|---|---|
| Chrome RSS ceiling (VPS unit) | `MemoryMax=900M`, `MemoryHigh` flip-flopped (650M→unset) | `deploy/vps/units/atlas-browser.service`, `atlas-browser-notes.md` |
| Chrome runtime behavior | wedged renderers, forced recycle every ~4 h, swap-in 20–35 MB/s → 1.4 MB/s post-fix | `atlas-browser-notes.md` |
| Operational cycle (when Chrome works) | 30 s heartbeat cadence; capture limited to one match per cycle by JS injection cost | `heartbeat.json`, runner |
| Current state | 0 priced selections since 2026-11-02 502-class failure | `heartbeat.json` |
| Browser-coupled LOC | ≈ 4,700 | §1.1 |
| Transport deps | httpx, pydantic already required; **no new deps** except gated curl_cffi | `pyproject.toml` |
| Resident memory of the new transport (process) | **NOT YET MEASURED** — expected one Python process ≈ 25–45 MB RSS; must be proven, not asserted | — |

### 10.2 Measurement plan (replaces estimates)

- Shadow phase: run DOM and API captures side-by-side for ≥ 200 cycles on the
  VPS; record per-cycle wall time, CPU%, and cgroup (`MemoryCurrent`/`Peak`)
  for the reader process and anything spawned.
- Accept targets (proposed): read cycle p95 ≤ 2 s; process RSS ≤ 60 MB;
  zero Chrome process when `--submit` is off; zero Chrome CPU share.
- Freshness: odds payload age sampled each cycle; assert ≤ 10 s live.
- Report in `docs/browserless-transport-design.md` §10 addendum — numbers, not
  adjectives. (Per directive: benchmark actual usage; theoretical estimates are
  not accepted as evidence.)

---

## 11. Migration plan (staged, reversible)

**Phase A — Reconnaissance (2–3 days).** §3 U1–U12; sanitized fixtures to
`tests/fixtures/api/`; `docs/api-recon.md` records findings; secrets rotated
per §7. *No runtime change.* Gate: every U1–U10 resolved or explicitly waived
by operator.

**Phase B — Transport foundation (2–3 days).** `atlas/transport/*` skeleton;
vault + redaction; `RealbetlistPage` schema validated against the existing
archives (offline replay — no network); contract tests green. Gate: fixture
replay tests pass; redaction test proves no cookie bytes in any log path.

**Phase C — Shadow operation (≥ 200 cycles / ≥ 7 days).** `ApiBoardSource`
runs alongside `DomCapture` (read-only, no decisions taken from API data).
Daily diff report: fixture coverage, market coverage, odds equality ratio,
freshness p50/p95, error taxonomy counts. Gate to proceed: odds exact-match
≥ 99% on priced families, market coverage ≥ 95% (B29 bar), zero unexplained
schema quarantines, session uptime ≥ 7 days without manual intervention.

**Phase D — Browserless read path.** Flip `ATLAS_TRANSPORT_READ=api`.
Balance/settle/cashout-reads move to API (settle first — it is the most
proven, `realbetlist` body already archived). DOM adapter retained, idle.
Campaign untouched: still consuming the same ledger, same gates. Rollback =
flip the flag (seconds).

**Phase E — Runtime elimination.** Remove the browser from the operational
startup (units: `atlas-browser`, `atlas-xvfb` ad-hoc only; settle timer no
longer needs a tab). Keep the DOM adapter for tests + parity harness; keep
actuation per §9 (default Option A: click-actuation on a minimal browser,
in-process or a separate cron-spawned throwaway instance). Sign-off
checklist: 100-cycle soak with live campaign wall closed; operator reviews
resource measurements; decision on actuation A/B taken separately.

Rollback at every phase = one env var. No phase deletes data, tests, or the
DOM adapter before Phase E sign-off.

---

## 12. Testing and acceptance criteria

**Keep:** all 536 existing tests, fixtures, CI. **Add:**

- Contract tests: every archived fixture validates against its pydantic schema;
  `bizCode != 10000` and `isAvailable=false` fixtures → correct typed errors.
- Session: vault round-trip; permissions check (`0600`); redaction test (no
  `sb_fs_id`/`usrId` in serialized errors/logs); expired-cookie fixture →
  `SessionExpiredError`; never-healthy-on-flaky-probe test.
- Transport: timeout→`PRE_SEND`/`OUTCOME_UNKNOWN` split (mocked); 429 with
  `Retry-After`; 5xx retried ≤2 for GETs and **never** for POSTs; malformed
  JSON → quarantine + `SchemaDriftError`; empty `entityList`/`totalNum=0` →
  `UnreadableBoard`; partial coverage < 95% → `PartialCoverage`.
- Staleness: synthetic aged payloads → `StaleDataError`; boundary TTL tests.
- Parity: `ApiCapture` vs `DomCapture` on captured board fixture
  (`fortuna_stuttgart_90plus.json` tape) → ≥95% coverage, identical
  normalization output on the overlap.
- Failure propagation: monkeypatched transport failure → `CaptureResult.ready
  == False`, decisions abstain, campaign stays `collect`/frozen, ledger gets no
  new exposure.
- Order reconciliation (design-only fixture tests for Option B): A accepted →
  VERIFIED via second-source fixture; A refused (200 + refusal envelope) →
  REJECTED, no ledger write; ambiguous timeout → OUTCOME_UNKNOWN, reconciliation
  by shortId lookup. **All skipped-by-default; no live calls.**

**Acceptance criteria for cutover (Phase D→E):**

| Criterion | Target |
|---|---|
| Shadow parity: odds exact-match (priced families) | ≥ 99% |
| Shadow parity: market coverage | ≥ 95% |
| Transport error rate (excluding session expiry) | < 1% of cycles |
| Session uptime between manual reseeds | ≥ 7 days (measured) |
| Read cycle p95 | ≤ 2 s |
| Process RSS | ≤ 60 MB |
| Unreadable-board events misclassified as empty board | **0** (hard gate) |
| Existing test suite | 536/536 green |
| Fail-closed soak | ≥ 100 cycles, wall never opened without VERIFIED settlement path |

---

## 13. Identified risks and unresolved engineering questions

| # | Risk / Question | Mitigation / Owner |
|---|---|---|
| R1 | **TLS/H2 fingerprinting**: SportyBet edge may reject non-browser JA3. Plain httpx could get 403/HTML. | Gated A/B replay experiment in Phase A (U10). If blocked: `curl_cffi` with `impersonate="chrome"` — pre-approved tier-2 by directive; measured, justified, and documented. No other deps. |
| R2 | `Outcomes` is signed/keyed (U3/U2) — replay fails outright | Fallback: keep board reads via a **local headless-free capture service**? Rejected — that reintroduces a browser. Real fallback: Option A-style minimal actuation/reading browser, *not* the full CDP rim. Decision point for operator before Phase C. |
| R3 | Endpoint drift (silent schema change) | archive-then-map + quarantine + nightly fixture replay in CI; drift alert via OTLP/notify |
| R4 | bizCode semantics undocumented; auth expiry may not map to 4xx | U7 capture on expired cookie; treat *any* ambiguity as expired (fail closed) |
| R5 | Session lifetime unknown; mid-cycle expiry | cheap probe each cycle (U5); expiry rate feeds heartbeat; reseed runbook |
| R6 | Legal/ToS on scripted access | operator decision; transport unchanged either way; rates ≈ SPA's own |
| R7 | Cookie vault on disk = concentrated secret | 0600/0700, no git, rotation advice, redaction tests |
| R8 | Direct order API tempting "because it's now easy" | **explicitly out of scope**; §9; needs separate approval + Option-B design |
| R9 | `data/ops/betsdetail_temp.json` holds live cookie + phone number | Phase A sanitizes into fixtures; recommend session/password rotation |
| R10 | VPS capacity for shadow phase | API reader is tiny; DOM path stays as-is until Phase D — no added pressure |

**Open questions for the operator (blocking Phase A):**

1. Actuation preference for Phase E: Option A (minimal click-browser) vs. defer
   the decision and keep status-quo placement for now?
2. Seeding method: OK to do a one-time `Network.getAllCookies` pull, or prefer
   manual cookie export (Option B)?
3. Is a staging/throwaway SportyBet account available for recon (preferred for
   the deliberately-expired-cookie probe, U7/U8), or recon on the live account?
4. Retention of the DOM adapter post-Phase-E: agree it stays as non-operational
   parity harness (recommended), or be deleted outright?
5. Rotation of the current session/password (per R9)?
6. Acceptance targets in §12 — any tightening desired (e.g. freshness ≤ 5 s)?

---

## 14. Challenges to this directive's assumptions

1. **"Eliminate Chrome entirely, no permanent fallback."** Agreed for the
   *read* path. Not agreed for deletion: the DOM adapter is the only
   ground-truth parity oracle for the API and the only tested placement path;
   it should remain in-repo as a *non-operational* harness after sign-off.
   "Not part of normal operating architecture" ≠ "deleted".
2. **"httpx as primary, curl_cffi only if experiments prove necessity."**
   Correct order, but note the asymmetry: if TLS fingerprinting is in play,
   httpx fails in a way that is hard to distinguish from a dead cookie without
   R1's A/B experiment. Build the experiment *first* (it is an afternoon), not
   after a failed week.
3. **"Do not introduce dependencies without measurable justification."**
   Accepted, with the same bar applied to the *removal* claim: the necessity of
   keeping any browser at all collapses to the actuator question (§9), which is
   a money-path decision deferred to the operator — not assumed away.
4. **Recon via Kimi daemon** — agreed to avoid as a dependency; CDP
   `Network.getResponseBody` suffices for read-only capture and is already the
   proven transport on this box.
5. **"Rewrite capture.py as the API client" is not proposed.** Rewriting
   `capture.py` would risk subtle behavior changes in the readiness/coverage
   semantics the engine implicitly depends on. Instead: additive
   `ApiCaptureSource` implementing the same surface, selected by flag.
6. **Priority order respected:** reliability > weight > determinism >
   complexity. Where they conflict (e.g., keeping DOM adapter vs. "minimal
   complexity"), reliability and reversibility win, per this directive's own
   ordering.
7. **The honest ceiling:** even perfect transport cannot fix edge quality.
   After Phase E, the binding constraint on Atlas is calibration sample volume
   (`MIN_CALIBRATION_OBS=50`, campaign n=5→100) — this redesign buys a reliable
   sensor, not a positive EV. Worth stating plainly so the win is not
   misread.

---

## Appendix A — File/evidence index

- `atlas/live/bet_history_scrape.py`, `data/ops/betsdetail_temp.json`
  (sanitized in doc §2.1–2.2), `data/ops/realbetlist_2026-10-28.json`
- `data/ops/net1.json`, `data/ops/netlist_temp.json`, `data/ops/nd.json`
- `atlas/live/capture.py`, `atlas/browser/*`, `atlas/live/runner.py`
- `atlas/ops/{ledger,campaign,settle_history,posture,boss}.py`, `atlas/live/decisions.py`
- `deploy/vps/units/atlas-browser.service`, `deploy/vps/atlas-browser-notes.md`
- `docs/{mission,engineering,bugs_and_issues}.md`, `README.md`
- `data/ops/heartbeat.json` (2026-11-03T14:30Z), `data/ops/campaign_state.json`
- `pyproject.toml` (536 test functions verified: `grep -c "def test" tests/*.py`)

---

## Appendix B — Implementation status (as of initial landing)

**Landed (this change):**

- `atlas/transport/errors.py`, `schemas.py` (Envelope/Realbetlist), `client.py`
  (sync httpx; pre-send/unknown/accepted taxonomy; GETs ≤2 retries on pre-send/5xx;
  POSTs never retried; 401/403 → SessionExpiredError), `session.py` (seeded vault,
  0600, redaction), `snapshots.py` (archive-then-map + quarantine),
  `sources/settle.py` (full realbetlist path → identical settle_state.json,
  parity-guarded vs `bet_history_scrape.parse_ticket`), `sources/balance.py`
  (envelope-tolerant recursive balance sniffer; refuses to guess),
  `sources/board.py` (**dump_raw recon only — parsing gated on Phase A fixtures**),
  `sources/cashout.py` (`cashable_count()` only; offer listing gated),
  `executor.py` (DirectOrderExecutor: refused without `allow_submit=True`,
  recipe-gated `ATLAS_ORDER_RECIPE`, no POST retries, verified → ledger
  `mark_verified_placed()` stamps the book's createTime epoch-ms as stake time),
  `verify.py` (independent confirmation via the realbetlist path — exact
  match on eventId/marketId/outcomeId + stake + odds inside a 15-min
  creation window; zero or many matches → None → accepted_unverified),
  `seed.py`, `session_probe.py`.
- Ledger: `BetRecord.placed_at_epoch_ms`; `place_bet()` stamps it; new
  `mark_verified_placed()` overwrites with the book's createTime and appends a
  `bet_verified` event. Additive only — existing consumers unaffected.
- Deleted: `data/ops/betsdetail_temp.json` (live cookie + phone in plaintext;
  useful header profile + tickets sanitized into `tests/fixtures/api/` first).

**Deliberately NOT landed / open:**

- Board parsing (`Outcomes` schema) — Phase A capture required; fabricating it
  would be fiction.
- Cashout offer parsing — Phase A capture required.
- Direct order execution is NOT wired into the runner, NOT reachable from any
  flag, default-refuses. Enabling = Phase A + operator sign-off + separate
  approval, per §9. The click path remains the only live actuator.
- Settle path via API is available (`python -m atlas.transport.sources.settle`)
  but the hourly `atlas-settle` unit is NOT cutover yet — it flips after one
  successful live shadow run against the DOM scraper's output.
- Existing 536 tests to remain green; new tests: `tests/test_transport_http.py`,
  `tests/test_transport_session.py` (parity + boundary coverage).

**Measured baselines still owed:** shadow-phase resource numbers (§10.2) —
not estimable honestly on this machine.

---

## Appendix C — Live verification log (2026-11-03)

Executed against the real site from the operator box (Windows CPython 3.13
`.venv-win` via WSL interop; Linux curl via WSL for raw HTTP baselines).
**No authenticated run was performed — the live session lives on the VPS and
the local Chrome profile is ACL-blocked to both the Windows user and root.**

Verified live:

| Check | Result |
|---|---|
| WAF/edge with bare curl (no client hints) | **HTTP 403** |
| WAF/edge with full Chrome 155 header profile | **HTTP 200** (`/ng/`, 99,605 bytes) |
| Guest read `sportList?option=1&productId=1` via new `TransportClient` | **200, bizCode 10000**, live sport menu (Football, Tennis, …) |
| Guest read `factsCenter/banned/events` | **200, bizCode 10000**, 461 banned fixture IDs |
| Guest read `flexiblebet/v2/getOddsKey` | **200, bizCode 10000** |
| Guest POST `factsCenter/Outcomes` | **403, `{"bizCode":19003,"message":"Forbidden","data":{}}`** — auth mandatory; captured as `tests/fixtures/api/outcomes_guest_forbidden.json` |
| Dummy-vault probe vs live `userBal/NGN` | **HTTP 401 → clean "session EXPIRED/REJECTED", probe exit 3** |
| Dummy-vault `sources.settle` vs live | **HTTP 401 → "re-seed the vault", exit 2** — no file written, no traceback |
| Junk POST through transport | **HTTP 500 → `OutcomeUnknownError`** (ambiguous, never auto-retried) |
| Windows-side DNS flakiness observed once; recovery confirmed | retry covered; WSL resolver unaffected |

Consequences:

1. **Header-profile scoring, not TLS, is the observable gate** — curl passed
   (200) with the full profile; bare curl does not. Plain `httpx` has now
   passed every gate exercised; `curl_cffi` remains unproven and therefore
   NOT introduced. Re-test if the edge starts JA3-scoring.
2. The guest reads confirm the transport client, request/response envelope
   discipline and fixture archiving all behave against the real book.
3. Authenticated reads (`Outcomes`, `realbetlist`, balance) are unverifiable
   until the vault is seeded on the operator/VPS side. The 401/403 paths
   are already proven fail-closed against the live site.

Steps remaining, in order, on the operator box/VPS:

1. One-time vault seed (either: CDP `Network.getAllCookies` pull on the
   logged-in VPS Chrome → `atlas.transport.seed --cookies cookies.json`, or
   a manual cookie export of the same).
2. `python -m atlas.transport.session_probe` → expect `ALIVE` + NGN balance.
3. `python -m atlas.transport.sources.settle --max-pages 2 --out data/ops/settle_state_http_shadow.json`
   and diff ticket sets against the DOM scrape in `settle_state.json`
   (count, order_ids, statuses). Do NOT overwrite the DOM state file.
4. Only then decide `atlas-settle` cutover; shadow mode for ≥7 days/≥200
   cycles before removing the browser from any operational path.

Interop note: running Windows CPython from WSL does NOT inherit WSL env vars
without `WSLENV`; use `cmd.exe /c "set VAR=... && ..."` (as used above) or
PowerShell for `$env:`.

## Appendix C.1 — 2026-11-03 16:09–16:13 UTC: fresh re-extract, endpoint matrix, cookie-metadata check

Executed on the production VPS (`/opt/atlas`), read-only GETs only, no code
changes, no Chrome interaction, no PII printed (counts and key names only).

1. **Re-extract + re-seed.** `Cookies` file `mtime = Nov 3 16:10 UTC`.
   Copy → decrypt (23 v10 values, 0 undecodable; 22 unique names) → seed vault
   → 21 non-empty cookies persisted (one whitespace-only dropped by seeder;
   duplicate name across hosts collapsed). Vault re-seeded 0600, temp exports
   deleted.
2. **Probe after re-seed** → still **HTTP 401** (`userBal/NGN`), exit 3.
3. **Endpoint matrix** (same `TransportClient`, no new code):

   | Endpoint | Result |
   |---|---|
   | `GET /api/ng/pocket/v1/finAccs/finAcc/userBal/NGN` | 401 |
   | `GET /api/ng/patron/account/info` | 401 |
   | `GET /api/ng/orders/order/v2/realbetlist?isSettled=10&pageSize=5&pageNo=1` | 401 |

   Uniform 401 across independent endpoints → rejects hypothesis §7.3
   (endpoint-specific rejection) and confirms §7.1/§7.2 (server-side session
   invalidation / lapsed short-lived access credential that the wedged SPA
   no longer re-mints).
4. **Cookie metadata** (names and expiries only; values never read):
   `userCert`, `refreshToken`, `userId`, `phone`, `selfExclusion*` all
   persistent and **not expired** at browser level (all expire 2026-12-03
   UTC). `sb_fs_id` already expired 2026-10-31 UTC; `sb_fs_flag` fine.
   Browser-level expiry is **not** the cause of the 401.
5. **Interpretation.** The on-disk jar is current and syntactically valid,
   yet the book rejects it on every authenticated route. The SPA re-mints
   the short-lived `userCert` from `refreshToken` while healthy; the wedged
   Chrome (load ~13, CDP 9222 mute) cannot, so the server-side session
   lapsed even though the cookies persist. This is an operator-facing outage,
   not a transport defect.
6. **Not done (deliberately):** no refresh-token→access-token exchange was
   attempted or probed (would be invented behaviour without operator
   approval and a dedicated recon); no browser action taken; settle shadow
   NOT run (gated on a 200); `atlas-settle` untouched; no files under
   `data/ops/` modified.
7. **Operator options:**
   a. Restore the VPS Chrome session (operator repairs/restarts Chrome or
      XVFB; Chrome currently wedged) and let the SPA refresh once — then
      re-extract + re-seed + probe (the recipe in §4 of the handoff works
      unchanged).
   b. Operator performs a manual login (headful or exported cookies) and
      re-seeds the vault from that export; probe to confirm.
   c. Only with explicit operator approval: dedicated recon of the
      `refreshToken` → access-credential exchange the SPA uses, and even
      then implemented as an opt-in, recon-verified step — never silently.
8. **Until one of the above lands:** the authenticated read path stays
   unverified; settle-shadow parity and Phase-A board recon remain blocked;
   the 401 fail-closed behaviour is the proven, correct response.

## Appendix C.2 — 2026-11-03 16:14–16:35 UTC: session restored, probe ALIVE, settle shadow parity 266/266

The operator restored the VPS Chrome session. Read-only CDP check: Chrome
156.0.8078.79 answering on 127.0.0.1:9222, one tab on a live SportyBet
football page, `Cookies` mtime 16:16:58 UTC. No Chrome action was taken by
this agent; no data was read from the browser beyond cookie names/expiries
and target-list paths.

1. **Re-extract + re-seed (16:17 UTC):** 24 v10 values decrypted, 23 unique
   names, 0 undecodable; vault re-seeded 0600; temp artifacts deleted.
2. **Probe → schema drift, not 401.** HTTP 200, `bizCode 10000`, envelope
   `{bizCode, message, data:{avlBal, status, avlCoins, coinsStatus,
   lastUpdatedTime, userFinAccNotExist}}`. The payload carried no key in
   `BALANCE_KEY_CANDIDATES`, so the source quarantined the raw body
   (`data/quarantine/balance/20261103T161807Z-fa169dd6.bin`) and raised
   `SchemaDriftError` — fail-closed as designed; nothing was guessed.
3. **Balance source fix (local + VPS copy of `sources/balance.py`):**
   `avlBal` added to the candidates with provenance note; `avlCoins` /
   `coinsStatus` deliberately NOT candidates (coins ≠ NGN balance).
   Fixture `tests/fixtures/api/user_bal_ngn.json` (synthetic values) + 2
   tests; full local suite **563 passed**. VPS copy synced; no tracked
   server files touched.
4. **Probe after fix → `session ALIVE — balance NGN 1119600.00 (via
   avlBal)`.** Unit of `avlBal` is UNVERIFIED: DOM heartbeat at 14:30Z
   read NGN 111.96; raw `avlBal` is 1119600.00 (ratio exactly 10⁴).
   No conversion factor has been invented; the NGN-denominated reading
   must be confirmed against the wallet UI before any money-path consumer
   relies on it. (The probe itself only feeds session health for now.)
5. **Settle shadow, full fetch (`--max-pages 8`):** **266/266 tickets,
   zero missing, zero extra; stake, winnings and status match 266/266
   (100.0%) vs the DOM oracle** `settle_state.json` (read-only; never
   overwritten). API `totalNum` = 266 equals the legacy DOM count. The
   shadow lives at `data/ops/settle_state_http_shadow.json`.
   **The API settle path is verified against the DOM oracle.**
6. **Not done / unchanged:** `atlas-settle` NOT cutover (operator decision;
   shadow-mode criteria — ≥7 days / ≥200 cycles — still apply before any
   browser removal); campaign wall, walls, stake caps, `operator_arm`
   untouched; no execution wiring; Phase-A Outcomes recon still open;
   nothing committed; repo remains dirty with the operator's pre-existing
   changes; VPS `data/ops/settle_state.json` unmodified.
7. **Open questions for the operator:**
   a. Confirm the true NGN wallet balance in the UI and the unit of
      `avlBal` (decide whether the probe should divide or report raw).
   b. Keep Chrome healthy for the shadow window, or re-seed cadence if it
      wedges again (the cookie jar outlived the SPA's re-minting).
   c. Whether to schedule `atlas-settle` shadow-mode review.
