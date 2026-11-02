<div align="center">

# ATLAS

### A closed-loop quantitative system for live football markets

*Priced by deterministic mathematics. Executed by browser automation.
Judged — without mercy — by its own ledger.*

<br>

`python 3.10+` · `pytest suite: green` · `submission: OFF by default` · `campaign: fail-closed`

</div>

---

Atlas is a machine for making **one** decision well and repeating it for hours:
watch a live football board, price every market from first principles, decide
whether the book is wrong by a wide enough margin to survive the truth, and —
only then — place a bounded, mathematically-sized stake in a real-money book.

Everything about Atlas is built around a refusal. It refuses to place a bet it
cannot price. It refuses a stake it cannot justify. It refuses certainty it
cannot earn. When the evidence says the edge is not real, it refuses to keep
playing — automatically.

> The most dangerous number in this system is a 1.0.
> No live football outcome is ever certain. Trust no model that claims otherwise.

---

## What it is · What it is not

**Atlas is —**

- A closed-loop live-betting execution system: capture → price → gate → stake
  → settle → recalibrate, running unattended against a real book, in NGN.
- An engineering study in honest probability: isotonic calibration, fractional
  Kelly sizing, pre-registered success criteria, and a verdict mechanism that
  can shut the whole operation down.
- A browser automation system with operator discipline: single-shot money
  path, read-only dashboard, watchdogged sessions, evidence on every action.

**Atlas is not —**

- A prediction machine. It claims no crystal ball. The price engine is
  deliberately **deterministic** — no neural networks, no black-box "AI picks".
- A profit rig with published performance claims. Its edge must be proven on
  its own ledger, at a pre-registered sample size, before money flow is
  permitted — and it is designed to freeze itself if that proof fails.
- Affiliated with any bookmaker. Not endorsed by, connected to, or operating
  on behalf of SportyBet.

---

## Architecture

One process, two planes. The fast plane is pure mathematics. The slow plane is
control: sessions, gates, ledgers, verdicts.

```text
  SportyBet live board                     ATLAS (one python process)
  ════════════════════╗   ┌──────────────────────────────────────────────────┐
                      ║   │                                                  │
  Chrome ◄── WebBridge ║  │  FAST PLANE · deterministic pricing              │
  extension  (daemon)  ║   │                                                  │
                      ║   │    board capture ── one evaluate-injection       │
      ▲               ║   │                     bundle read, single round    │
      │ snapshots /   ║   │                           │                      │
      │ clicks / fills║   │                           ▼                      │
      │ money path    ║   │    de-vig ── proportional · power · Shin          │
      │ (single-shot) ║   │                           │                      │
      │               ║   │                           ▼                      │
                      ║   │    clock/stoppage model ── remaining minutes      │
                      ║   │                           │                      │
                      ║   │                           ▼                      │
                      ║   │    Poisson / Dixon–Coles posterior score matrix   │
                      ║   │                           │                      │
                      ║   │                           ▼                      │
                      ║   │    every market priced off ONE joint matrix       │
                      ║   │    (1X2 · totals · handicaps · BTTS · paths …)    │
                      ║   │                           │                      │
                      ║   │                           ▼                      │
                      ║   │    isotonic (PAVA) calibration · per family       │
                      ║   │                           │                      │
                      ║   │                           ▼                      │
                      ║   │    fractional Kelly sizing, hard floor and cap    │
                      ║   │                           │                      │
                      ╚═══╪═══════════════▼══════════════════════════════════┤
                          │  SLOW PLANE · control                            │
                          │                                                  │
                          │    risk gates · campaign wall · ledger · verdict │
                          │                           │                      │
                          │                           ▼                      │
                          │    fill ──► upload ──► ONE shot, then evidence    │
                          └───────────────────────────┬──────────────────────┘
                                                      │
                          Groq LLM (supervisor) ◄─────┘
                          exception recovery ONLY · zero money
                          tools · zero pricing authority
```

The separation is absolute and deliberate:

- **The mathematical engine is the only source of price.** No LLM ever sees,
  adjusts, or overrides a probability.
- **The LLM never touches money.** It exists for the one job automation is
  worst at — recovering from unpredictable site failures — and holds no
  placement authority of any kind.
- **The money path is single-shot.** `click`/`fill`/`upload` run once. Ambiguity
  is resolved by evidence, never by retry. A second click can be a second bet.

---

## The pricing engine

Live in-play markets give you a scoreboard and a book. Atlas derives fair
probabilities from first principles, in real time:

1. **De-vig the board.** Three independent models — proportional, power, and
   Shin — strip the bookmaker's margin out of the offered prices. Where the
   three disagree, the disagreement is treated as information, not noise.
2. **Estimate the clock played.** A stoppage-aware model converts the visible
   match clock into effective remaining minutes, correcting for inflation from
   injury time.
3. **Infer attack rates.** Pre-match λ's are recovered from moving odds; live
   statistics (shots, corners, cards, xG) update the posterior.
4. **Build the joint score matrix.** A Poisson/ Dixon–Coles grid (with τ
   correlation correction) over remaining scorelines becomes the world model.
5. **Price every market off one matrix.** 1X2, totals, handicaps, BTTS,
   odd/even, double chance, race-to, goals-in-row, Asian lines and draw-no-bet
   are all marginal/combinatorial reads off the same joint distribution —
   no ad-hoc per market models, no possibility of internal contradiction.
6. **Compute EV, honestly.** Every decision carries `model_p`, `calibrated_p`,
   `ev`, and `ev_lower`. Acceptance uses the *low* estimate, not the flattering
   one.

### Why calibration exists (the honest part)

An early live window produced the numbers that motivated this layer. Across
that window the ledger's claimed edge averaged +13.1% while realised return
came in near +9.5% — a disappointment, within noise. The indictment was more
specific: one over-goals family had priced roughly +8.9% of claimed EV and
returned −38.3% realised. Small sample, one family, one window — which is
exactly why the response is statistical, not anecdotal.

Atlas fixes this without deleting history. It re-linearises it: an
**isotonic (PAVA) calibration map is fit per market family from settled
outcomes**, mapping raw `model_p` to an empirically-corrected `calibrated_p`
— and it is the calibrated number, not the raw one, that gates EV. Families
without at least 50 settled observations are *not* calibrated; they trade on
shrinkage-adjusted raw probabilities and say so, loudly, in the decision record.

Every hour, a scorer rewrites the campaign verdict from the ledger:

```json
{
  "action": "collect",
  "why": "settled n=0 < MIN_N=100",
  "min_n": 100,
  "cutoff": "2026-11-01T12:00:00"
}
```

Every threshold lives in configuration. Every label lives in the ledger. Every
conclusion — including "we cannot know yet" and "this family is broken" — is a
first-class output.

---

## Sizing: fractional Kelly, bounded

Once a selection survives every gate, stake size is not a mood. It is a
fraction (0.10) of the Kelly-optimal stake computed off the *calibrated*
probability and the observed price, bounded by NGN floor, NGN cap, per-match
exposure cap, and the campaign flat unit when the operator is silent.

Kelly on a miscalibrated p is a money shredder. Kelly on a *calibrated* p,
fractionally discounted, with hard floors and caps — that is a defensible
algorithm. The NGN floor keeps the stake survivable; the support cap keeps the
campaign's statistical footprint meaningful.

---

## The risk gates

Every candidate must pass every gate. The refusal reasons are recorded
word-for-word, so the ledger doubles as a forensic record of what the system
thought and why it said no.

| Gate | Threshold | Meaning of the line |
|---|---|---|
| Minimum EV | ≥ 0.05 | Below 5% edge the book's noise will eat you — no trade. |
| EV ceiling | ≤ 0.25 | An EV too good to be true usually is. Something is mispriced upstream; do not touch it. |
| Certainty ceiling | p < 0.95 | A model claiming ≥ 95% probability of any in-play football outcome is broken — abstain. p = 1.0 is always refused. |
| Odds ceiling | ≤ 3.0 | At high decimal odds a 1% probability error is the whole stake. The NGN 100 loss on @10.00 was the case study. |
| Probe filter | — | Selection names that smell like test/debug artefacts are refused outright. |
| Duplicate cap | 1 per selection | One position per selection. No averaging down. No re-entry. |
| Exposure cap | per-match & global | Correlated bets on one match cannot stack beyond the cap. |
| Family stop-loss | 3 consecutive settled losses | A market family that keeps losing freezes itself until a win resets the streak. |
| Campaign wall | `proven` only | **Placement is fail-closed.** Until the campaign record says "proven", no money leaves — collect / watch / observe / missing state / scorer errors all freeze before the slip. |

The campaign is pre-registered. Success criteria were fixed *before* capital
was deployed, and the verdict is recomputed from the ledger every hour:

| Verdict | Criterion | Consequence |
|---|---|---|
| `collect` | fewer than 100 settled bets | observe only · no placement |
| `watch` / `observe` | sample growing, proof not yet hard | observe only · no placement |
| `proven` | ROI confidence interval excludes zero **and** CLV ≥ 0 | gates open, within all caps |
| anything else | missing state, stale scorer data, scoring error | **fail-closed** · no placement |

> The pre-registration is the point. Atlas decided what success would look like
> *before* deploying capital, then lets the settled outcomes speak — and binds
> the machine to whatever they say. A freeze is not a failure of the system;
> it is the system working.

---

## Browser automation with operator discipline

The machine drives a real browser — because that is what the venue is. Chrome
logs into the target, WebBridge's daemon + extension execute a small, strict
command set; Atlas orchestrates.

| Layer | What runs there |
|---|---|
| Atlas loop (`atlas/live/runner.py`) | Capture → price → gate → place → settle, forever, with a watchdog and a heartbeat. |
| WebBridge daemon (`127.0.0.1:10086`) | `/command` navigate · snapshot · click · fill · upload · list_tabs, session-scoped (`agent:atlas-live`), Chrome-extension served. |
| Capture (`atlas/live/capture.py`) | One evaluate-injection bundle per cycle — board parsed in ~a dozen seconds, not minutes. Read timeouts of 5s so a wedged page fails fast instead of stalling the loop. |
| Money path (`atlas/browser/betslip.py`) | The bet slip interactions — click/fill/upload — are single-shot. A failed attempt is *evidence*, never a retry signal. |
| Supervisor (`atlas/live/supervisor.py`) | Groq LLM, only invoked on exceptions: re-login, tab recovery, session rescue. No pricing, no stake, no placement authority. Zero money tools. |
| Watchdog (oneshot timer) | Kills and recovers Chrome / WebBridge daemon liveness failures before the operator notices. |

No headless masquerade, no shadow scraping: Atlas drives the real logged-in
session — the same account you would drive yourself — at a pace you can watch,
with every action logged.

---

## Where the AI is allowed — and is not

Atlas is an AI system that decided where AI doesn't belong.

- **No neural networks in pricing.** A live football price needs a defensible
  deterministic derivation at cycle speed. Statistical inference is: models
  (Poisson, Dixon–Coles, Shin de-vig, isotonic maps), not opaque learned
  approximations that drift silently under regime change.
- **The LLM has no authority over money.** It recovers a wedged session by
  re-authenticating it. It never proposes, modifies, or approves a stake. It
  does not see the money path. The wall is architectural, not advisory.
- **No LLM-generated telemetry.** Every number in the ledger comes from the
  ledger itself — from the game, the clock, the board, or the settlement.

---

## Evidence: the ledger is the product

Everything decision-shaped is recorded on append-only JSONL:

- `bet_placed` — what, when, at what price, at what calibrated probability, at
  what stake, at what gate state
- `bet_settled` — outcome, P&L, the realised-vs-claimed delta
- `decision` & `gate refusal` records — the reasoning at each gate, including
  all failures

The campaign scorer reads the ledger and prints the verdict. The operations
dashboard (`http://…/status`) is a **read-only observer**: heartbeat liveness,
exposure split (recent / unreconciled), posture state, and the one thing that
matters — the campaign gate state. It can slow posture, but it cannot force a bet.

Records are versioned, checksummed, and retained. Telemetry (OTLP) exists, but
is optional — evidence is designed to survive without any observability stack.

---

## Quickstart

```bash
git clone https://github.com/EmmanuelAdesina/atlas.git
cd atlas
pip install -e ".[test]"

# Run the regression suite (deterministic; includes all the case studies below)
python3 -m pytest -v

# Replay a recorded live observation through the full pricing engine
python3 -m atlas.main live-demo \
  --fixture tests/fixtures/fortuna_stuttgart_90plus.json \
  --log data/decisions/demo.jsonl

# Post-settlement audit + calibration forensics from a decision log
python3 -m atlas.main boss \
  --fixture tests/fixtures/fortuna_stuttgart_90plus.json \
  --log data/decisions/boss_audit.jsonl

# Forecasts + posture + campaign state, read-only
python3 -m atlas.main run --dashboard --port 8077
```

The loop expects:

- a real SportyBet NG browser session (credentials are supplied only via env
  vars — `ATLAS_SPORTYBET_PHONE`, `ATLAS_SPORTYBET_PASSWORD` — never in code),
- a WebBridge daemon reachable at `ATLAS_WEBBRIDGE_DAEMON_URL`,
- optional LLM credentials for the exception-recovery supervisor only
  (`ATLAS_BOSS_BASE_URL`, `ATLAS_BOSS_MODEL`, `ATLAS_BOSS_API_KEY`).

**Nothing is placed until you explicitly flip submission on.** The default
operational posture is observe-and-log; real money requires explicit intent,
a proven campaign record, and every gate green.

---

## Case studies: failures taught us the gates

These are real incidents, in order, and the engineering responses. None are
hypothetical; all are grounded in the gate set you can read in the ledger.

1. **The certainty catastrophe.** Model claimed p = 1.0 on markets at 10.0×
   and 13.5× decimal odds. The stake was defensible only if the claim was true
   — it was not. **Response:** a hard probability ceiling (`p < 0.95` refusal)
   and a certainty gate that treats any 1.0 as an emergency, not a bet.
2. **Long-odds poisoning.** Flat NGN 100 stakes landed on lottery-shaped
   prices — a correct-score at 57.0×, an over-line at 10.0× — where the entire
   edge claim collapses if the probability model is wrong by a single
   percentage point. It was. **Response:** a decimal-odds ceiling. Calibration
   can correct the *centre* of a probability estimate; it cannot defend a stake
   at odds where the error bars are the size of the decision.
3. **The repeat-loss loop.** The same selection was re-entered after
   losing when the first attempt had already gone against it. **Response:**
   single-shot money path with no retries, one-position-per-selection cap,
   and a fail-closed campaign wall that freezes the entire book when the
   record doesn't prove the edge.
4. **The calibration gap.** While a family was uncalibrated, it traded on raw
   model probabilities that had never faced a settlement sample. **Response:**
   families with fewer than 50 settled observations are flagged as
   uncalibrated — they trade on shrinkage-adjusted raw probabilities, and
   every decision record says so explicitly.

Each gate exists because a real loss earned it. The refusal log quotes its own
gate parameters — every *why* a bet didn't happen is in the ledger forever.

---

## Repository

```
atlas/
├ atlas/live/           the two planes: math_engine (pricing), capture,
│                       estimate, decisions (gates), calibration, runner,
│                       control (dashboard), supervisor, boss
├ atlas/ops/            ledger, campaign verdict, posture, portfolio, settle
│                       + settle_history (reconciliation), telemetry, notify
├ atlas/browser/        WebBridge adapter: betslip, slipclear, cashout
├ atlas/collectors/     SportyBet board collection, Polymarket reference prices
├ atlas/models/         canonical domain models (sport, market)
├ atlas/mappers/        venue → canonical → venue
├ atlas/config/         settings, environment
├ atlas/http/           retry-safe HTTP client
├ tests/                the fixtures, the case studies, the guarantees
├ docs/                 mission · engineering · known issues (kept honest)
└ data/                 runtime state (gitignored): ledger, calibration maps,
                        campaign verdict, heartbeats
```

---

## Design invariants

1. **Submission is OFF until it is explicitly ON.** No code path enables real
   placement silently.
2. **The campaign wall is fail-closed.** Missing state = no money moved.
3. **One selection, one position.** Forever.
4. **p = 1.0 is a bug report, not a trade.**
5. **Never bets what it cannot price.** Unknown markets are refused, not
   guessed.
6. **The money path does not retry.** Ever. Evidence replaces attempts.
7. **The LLM recovers sessions, never prices markets, never moves money.**
8. **The ledger survives everything.** Append-only JSONL; the scoreboard of
   truth is what settled.
9. **"Unknown" is a valid result.** The system returns "unattributable"
   instead of manufacturing causal stories.
10. **Negative results are results.** A frozen campaign is a working outcome.

> The question Atlas was built around: *when the system's assumptions are
> wrong, what does it do?* Every door above is the answer to that question.

---

## Documentation

- [`docs/mission.md`](docs/mission.md) — the objective and non-negotiables
- [`docs/engineering.md`](docs/engineering.md) — architecture decisions and
  mathematical invariants
- [`docs/bugs_and_issues.md`](docs/bugs_and_issues.md) — the honest, open list

---

## Disclosure & responsibility

- Atlas drives a personal betting account with small, capped, self-limiting
  stakes. It is not financial advice, not a prediction service, and not a
  product for resale.
- Sports betting is a negative-expected-value entertainment market at the
  retail level; the entire construction of this project assumes the book is
  usually right and demands proof before trusting it to be wrong.
- Use responsibly, within local regulation. When in doubt, the system is
  designed — deliberately — to do the boring thing: nothing.

---

<div align="center">

### Atlas

*Built by an engineer who trusts gates more than opinions, ledgers more than
claims, and calibration more than convictions.*

</div>
