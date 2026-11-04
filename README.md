<div align="center">

# ATLAS

### A closed-loop real-time pricing engine for perishable contracts

*Deterministic pricing. Hard risk gates. An evidence ledger that outlives opinions.*

<br>

`python 3.10+` | `pytest suite: green` | `submission: OFF by default` | `fail-closed by construction`

</div>

---

Atlas prices contracts that resolve to a settlement within hours, against a
counterparty that re-quotes every few seconds, on a venue that exposes no
API: the market exists inside a browser page. The engine works the way a
market maker reads a book. Compute fair value from first principles, compare
it to the quote, decline almost every trade.

Most of the code exists to say no. Risk enforcement is where this project
spent its effort, and the refusal reasons are recorded as carefully as the
prices. When the evidence says the edge is not real, the system stops on its
own, before the money moves, without an operator in the loop.

> The most dangerous number in Atlas is 1.0. No live outcome is ever certain.
> A model that reports p = 1.0 is reporting a defect, not a probability.

---

## What it is, and what it is not

**Atlas does:**

- Derive prices for every contract family from a single joint posterior, so
  two quotes it prices differently are structurally impossible.
- Learn its own error rate: an isotonic calibration map per contract family,
  refit from settled outcomes.
- Size each position with fractional Kelly, inside hard floors and caps.
- Record every decision, every refusal, and every settlement in an append-only
  ledger, and recompute its verdict from that ledger hourly.
- Freeze itself when the pre-registered proof does not arrive.

**Atlas does not do:**

- Neural networks in pricing. The pricing core is deterministic by choice:
  every number downstream has a derivation you can audit.
- Published performance claims. The edge must be proven on the ledger at a
  pre-registered sample size before execution is permitted.
- Shadow scraping. It drives a real logged-in session, in the open, with
  every action logged.

---

## Architecture

One process, two planes. The fast plane is a deterministic pricing kernel. The
slow plane is control: sessions, gates, ledgers, verdicts.

```text
      live venue session                     ATLAS (one python process)
      ═══════════════════   ┌──────────────────────────────────────────────────┐
                       ┌──► │                                                  │
  Chrome ◄── WebBridge ║    │  FAST PLANE · deterministic pricing              │
  extension  (daemon) ║     │                                                  │
                       ║     │    market capture ── one evaluate-injection      │
      ▲                ║     │                     bundle read, single round    │
      │ snapshots /    ║     │                           │                      │
      │ clicks / fills ║     │                           ▼                      │
      │ execution path ║     │    de-vig ── proportional · power · Shin          │
      │ (single-shot)  ║     │                           │                      │
      │                ║     │                           ▼                      │
                       ║     │    clock/stoppage model ── remaining time         │
                       ║     │                           │                      │
                       ║     │                           ▼                      │
                       ║     │    joint posterior matrix over the remaining      │
                       ║     │    state space (Poisson / Dixon-Coles)            │
                       ║     │                           │                      │
                       ║     │                           ▼                      │
                       ║     │    every contract priced off ONE joint matrix     │
                       ║     │    (marginals · combinators · path contracts)     │
                       ║     │                           │                      │
                       ║     │                           ▼                      │
                       ║     │    isotonic (PAVA) calibration · per family       │
                       ║     │                           │                      │
                       ║     │                           ▼                      │
                       ║     │    fractional Kelly sizing, floor and cap         │
                       ║     │                           │                      │
                       ╚═════╪═══════════════▼══════════════════════════════════┤
                             │  SLOW PLANE · control                            │
                             │                                                  │
                             │    risk gates · campaign wall · ledger · verdict │
                             │                           │                      │
                             │                           ▼                      │
                             │    fill ──► upload ──► ONE shot, then evidence    │
                             └───────────────────────────┬──────────────────────┘
                                                         │
                             Groq LLM (supervisor) ◄─────┘
                             exception recovery only. no execution
                             tools, no pricing authority.

  deployed as systemd units:  atlas-loop · atlas-dashboard · atlas-browser
                              atlas-watchdog (oneshot) · atlas-settle.timer
```

Three rules make the split mean something:

- The pricing kernel is the only source of price. No LLM reads or modifies a
  probability.
- The LLM never executes. It exists for the one job automation handles badly:
  recovering a session after an unpredictable page failure. It holds no
  pricing or execution authority, and it cannot see the execution path.
- The execution path is single-shot. `click`, `fill`, `upload` run once.
  Ambiguity is resolved by evidence, never by retry, because a second click
  can be a second position.

---

## Pricing: one posterior, every contract

The feed gives you a clock, a set of state variables, and quotes. The engine
derives fair value in the following order:

1. **Remove the quoted margin.** Three independent de-vig models, proportional,
   power, and Shin, strip the overround out of the offered prices and produce
   a margin-free implied distribution. When the three disagree, the
   disagreement is recorded as information rather than averaged away.
2. **Estimate elapsed time.** A hazard-aware model converts the displayed
   clock into effective remaining time, correcting the inflation a venue puts
   into its countdown display.
3. **Infer the latent parameters.** Ex-ante rates are recovered from quoted
   prices. Live state variables update them as a posterior, with
   empirical-Bayes shrinkage on thin inputs.
4. **Build one joint posterior.** A Poisson/Dixon-Coles matrix, with the tau
   correlation correction, over the remaining state space.
5. **Price every contract off that one matrix.** Linears, totals, spreads,
   props, path and elimination contracts, and their combinations are all
   marginal or combinatorial reads of the same distribution. There are no
   per-contract models, so the engine cannot hold two prices that disagree
   with each other.
6. **Compute EV with a lower bound.** Every decision carries `model_p`,
   `calibrated_p`, `ev`, and `ev_lower`. Acceptance uses the low estimate.

A mispriced contract is survivable. Two internally contradictory prices are
not: that is one position hidden inside another, and the single-matrix rule
removes the failure mode instead of monitoring it.

---

## Why calibration exists

The first live window produced the numbers that justify this layer. Claimed
edge averaged +13.1% across that window while realised return came in near
+9.5%, a gap small enough to sit inside noise. One contract family had priced
about +8.9% of claimed EV and returned -38.3% realised. Neither number proves
anything on its own. Together they describe the failure a pricing engine is
built to catch, and the response is statistical rather than anecdotal.

Atlas does not delete that history. It recalibrates against it. An isotonic
(PAVA) map is fit per contract family from settled outcomes and maps raw
`model_p` to `calibrated_p`. The EV gates act on the calibrated number.
Families with fewer than 50 settled observations are not calibrated at all;
they trade on shrinkage-adjusted raw probabilities, and every decision record
says so.

An hourly scorer rewrites the campaign verdict from the ledger:

```json
{
  "action": "collect",
  "why": "settled n=0 < MIN_N=100",
  "min_n": 100,
  "cutoff": "2026-11-01T12:00:00"
}
```

Thresholds live in configuration. Labels live in the ledger. Conclusions
include the unflattering ones: "not enough data yet" and "this family is
broken" are both valid outputs.

---

## Sizing: fractional Kelly, bounded

Stake size is arithmetic, not judgment. A fraction (0.10) of the Kelly-optimal
stake, computed from the calibrated probability and the observed quote,
bounded by a floor, a per-position cap, an exposure cap per object, and the
campaign flat unit when the operator is silent.

Kelly sizing on a miscalibrated probability loses money faster than flat
staking does. On a calibrated probability, held to a small fraction of full
Kelly, it is a defensible algorithm.

---

## The risk gates

Every candidate must pass every gate. The refusal reasons are recorded
verbatim, which turns the ledger into a forensic record of what the system
thought and why it said no.

| Gate | Threshold | Rule |
|---|---|---|
| Minimum EV | ≥ 0.05 | Below 5% edge, quote noise eats it. No trade. |
| EV ceiling | ≤ 0.25 | Edge too good to be true usually is. Something is wrong upstream. Do not touch it. |
| Certainty ceiling | p < 0.95 | A probable-95%-or-better claim for any live outcome means the model is broken. Abstain. A recorded 1.0 is always refused. |
| Quote ceiling | ≤ 3.0 | At long odds a 1% probability error is the whole stake. The 100-unit loss at 10.0x paid for this line. |
| Probe filter |  | Inputs that look like test artifacts are refused outright. |
| Duplicate cap | 1 per selection | One position per selection. No averaging down, no re-entry. |
| Exposure cap | per-object and global | Correlated positions cannot stack past the cap. |
| Family stop-loss | 3 consecutive settled losses | A family that keeps losing freezes itself until a win resets the streak. |
| Campaign wall | `proven` only | Execution is fail-closed. Until the verdict says "proven", no order leaves. collect, watch, observe, missing state, and scorer errors all freeze before the slip. |

The campaign is pre-registered. Success criteria were fixed before capital
was deployed, and the verdict is recomputed from the ledger hourly:

| Verdict | Criterion | Consequence |
|---|---|---|
| `collect` | fewer than 100 settled positions | observe, no execution |
| `watch` / `observe` | sample growing, proof not hard yet | observe, no execution |
| `proven` | ROI confidence interval excludes zero, and CLV ≥ 0 | gates open within all caps |
| anything else | missing state, stale scorer data, scoring error | fail-closed, no execution |

Pre-registration is what makes the verdict binding. The success criteria were
written before capital was deployed, and the machine is bound to whatever the
settled outcomes say. A freeze is not a system failure. It is the system
passing its own test.

---

## Deployment and operational discipline

This is a deployed service stack with an opinion about every failure mode,
not a script that gets run when someone remembers to run it.

| Component | Discipline |
|---|---|
| systemd unit topology | `atlas-loop`, gated by an arm flag: no flag, no loop. `atlas-dashboard`, `atlas-browser`, `atlas-watchdog` (oneshot, timer-driven), `atlas-settle.timer` (hourly reconciliation). |
| Capture hot path | One evaluate-injection bundle per cycle. A full board parse takes on the order of twelve seconds. Read timeouts of 5 s fail fast on a wedged page instead of stalling the loop. |
| Execution path | Single-shot, no retries, one position per selection. Every attempt leaves evidence, successful or not. |
| Session liveness | Heartbeat freshness gates submission. A stale heartbeat flips the dashboard to not-submitting by construction, not by alarm. |
| Wedge recovery | A oneshot watchdog timer recovers wedged browser or daemon processes. |
| Posture steering | Risk posture (EV floor, Kelly fraction, caps) is configuration. The dashboard can tighten it and never loosen it. The ratchet only turns one way. |
| Reconciliation | A settlement timer reconciles the venue's own history against the ledger hourly. Missing settlements are surfaced, never assumed. |

No headless masquerade and no shadow scraping. The engine drives the real
logged-in session, the same account a human would drive, at a pace that can
be observed, and every action lands in the log.

---

## Where the AI is allowed, and where it is not

- **Pricing uses no neural networks.** Live prices need derivations that hold
  up at cycle speed: de-vig models, Poisson/Dixon-Coles posteriors, isotonic
  maps. Learned approximations drift silently under regime change, and a
  drifted price has no error bar you can see.
- **The LLM has no authority over money.** It re-authenticates a wedged
  session. It never proposes, modifies, or approves a position, and it does
  not see the execution path. The wall is architectural, not advisory.
- **No LLM-generated telemetry.** Every number in the ledger comes from the
  observable state, the clock, the quotes, or the settlement.

---

## Evidence: the ledger is the product

Everything decision-shaped is recorded on append-only JSONL:

- `bet_placed`: what, when, at what quote, at what calibrated probability, at
  what stake, in what gate state
- `bet_settled`: outcome, P&L, realised versus claimed
- decision and refusal records: the reasoning at each gate, including all
  failures

The campaign scorer reads the ledger and prints the verdict. The operations
dashboard (`http://…/status`) is a read-only observer: heartbeat liveness,
exposure split between recent and unreconciled positions, posture state, and
the campaign gate state. It can tighten posture. It cannot force an order.

Records are versioned, checksummed, and retained. OTLP telemetry is optional.
The evidence is designed to survive with no observability stack attached.

---

## Quickstart

```bash
git clone https://github.com/EmmanuelAdesina/atlas.git
cd atlas
pip install -e ".[test]"

# Full regression suite. The case studies below exist as tests.
python3 -m pytest -v

# Replay a recorded live observation through the full pricing engine
python3 -m atlas.main live-demo \
  --fixture tests/fixtures/fortuna_stuttgart_90plus.json \
  --log data/decisions/demo.jsonl

# Post-settlement audit plus calibration forensics from a decision log
python3 -m atlas.main boss \
  --fixture tests/fixtures/fortuna_stuttgart_90plus.json \
  --log data/decisions/boss_audit.jsonl

# Read-only operations dashboard: posture, gates, campaign state
python3 -m atlas.main run --dashboard --port 8077
```

Deployment notes:

- Venue credentials live only in environment variables, never in code. See
  `atlas/config/settings.py`.
- A WebBridge daemon must be reachable at `ATLAS_WEBBRIDGE_DAEMON_URL`.
- The exception-recovery supervisor takes optional LLM credentials:
  `ATLAS_BOSS_BASE_URL`, `ATLAS_BOSS_MODEL`, `ATLAS_BOSS_API_KEY`.

Nothing executes until the arm flag exists and submission is explicitly
enabled. The default posture is observe and log. Real money additionally
requires a proven campaign record and every gate green, and the campaign wall
enforces that at the last step before execution, failing closed.

---

## Case studies

These are real incidents from the decision and placement logs, in
chronological order, with the response each one earned.

1. **Certainty claims.** The engine recorded p = 1.0 on contracts quoted at
   10.0x and 13.5x. A stake is only defensible if that claim is true, and it
   was not. **Response:** a hard ceiling at p < 0.95. Any recorded 1.0 is
   treated as a defect to fix, never as an entry signal.
2. **Long-odds poisoning.** Flat 100-unit stakes at 57.0x and 10.0x. At those
   magnitudes a single point of probability error is the whole stake, and the
   error arrived. **Response:** a decimal-odds ceiling. Calibration can
   adjust the center of an estimate. It cannot rescue a stake where the error
   bars and the decision are the same size.
3. **Repeat entry.** The same selection was re-entered after the first
   attempt had already gone against it. **Response:** single-shot execution
   with no retries, one position per selection, and the fail-closed campaign
   wall.
4. **Uncalibrated trading.** Some contract families traded before any
   settlement sample existed for them. **Response:** fewer than 50 settled
   observations now flags a family as uncalibrated. Those families size off
   shrinkage-adjusted raw probabilities and say so in every decision record.

Every gate in the tables above was paid for with a loss.

---

## Repository

```
atlas/
├ atlas/live/           the two planes: math_engine (pricing), capture,
│                       estimate, decisions (gates), calibration, runner,
│                       control (dashboard), supervisor, boss
├ atlas/ops/            ledger, campaign verdict, posture, portfolio, settle
│                       settle_history (reconciliation), telemetry, notify
├ atlas/browser/        WebBridge adapter: slip, slipclear, cashout actions
├ atlas/collectors/     venue adapters: board collection, cross-venue reference
├ atlas/mappers/        venue to canonical, canonical back to venue
├ atlas/models/         canonical domain models
├ atlas/config/         settings, environment
├ atlas/http/           retry-safe HTTP client
├ tests/                fixtures, case studies, and the guarantees they bought
├ docs/                 mission, engineering notes, known issues
└ data/                 runtime state (gitignored): ledger, calibration maps,
                        campaign verdict, heartbeats
```

---

## Design invariants

1. Execution stays off until it is turned on explicitly. No code path enables
   real flow silently.
2. The campaign wall fails closed. Missing state means no money moved.
3. One position per selection, permanently.
4. A recorded p = 1.0 is a defect report, not a trade.
5. Nothing is entered that cannot be priced.
6. The execution path never retries. Evidence replaces attempts.
7. The LLM recovers sessions. It does not price, size, or execute.
8. The ledger is append-only, and the settlement is the source of truth.
9. "Unknown" is a valid result. The system reports what it cannot attribute
   instead of inventing a story.
10. A frozen campaign is a passing outcome.

The question the whole design answers: when the system's assumptions are
wrong, what does it do?

---

## Documentation

- [`docs/mission.md`](docs/mission.md): the objective and non-negotiables
- [`docs/engineering.md`](docs/engineering.md): architecture decisions and
  mathematical invariants
- [`docs/bugs_and_issues.md`](docs/bugs_and_issues.md): the open list of
  operational issues, kept current
- [`docs/browserless-transport-design.md`](docs/browserless-transport-design.md):
  design + rollout of the direct-HTTP (browserless) transport for SportyBet.
  Read sensors work today (`python -m atlas.transport.session_probe`,
  `python -m atlas.transport.sources.settle`); the board read is gated on
  Phase A recon fixtures. Direct order submission exists only as the disabled
  `DirectOrderExecutor` (see design §9) — it is not wired into the runner.

## Browserless transport (experimental)

SportyBet has no public API, but its SPA is cookie-authenticated JSON over
HTTP. `atlas/transport/` talks to it directly: a seeded cookie vault
(`python -m atlas.transport.seed --cookies cookies.json`), explicit timeout/
retry discipline (GETs idempotent, POSTs never retried), raw-response
archiving before parsing, and typed failures (`SessionExpiredError`,
`OutcomeUnknownError`, `SchemaDriftError`, `UnreadableBoardError`) that keep
an unreadable book from ever looking like an empty one. Settlement reads
(`realbetlist`) already produce byte-identical `settle_state.json`. Ledger
records carry `placed_at_epoch_ms`, and bets verified against Bet History are
stamped with the book's own `createTime` — the exact moment the stake left
the account. Chrome remains only as the actuator for live placement until
the design's Phase A–E gates pass.

---

## Disclosure and responsibility

Atlas operates a research account with small, capped stakes at a live
real-money venue. It is not financial advice, it is not a prediction service,
and it is not a product for resale.

The construction assumes the counterparty is usually right, and requires
pre-registered evidence before it allows that assumption to be disputed. Use
responsibly and within local regulation. When the system is in doubt, it does
the conservative thing, which is nothing.
