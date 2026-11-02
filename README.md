<div align="center">

# ATLAS

### A closed-loop real-time pricing engine for perishable, adversarial markets

*Posterior prices from first principles. Decisions gated by mathematics, not
opinions. Executed by disciplined browser automation on production infrastructure.*

<br>

`python 3.10+` · `pytest suite: green` · `submission: OFF by default` · `fail-closed by construction`

</div>

---

Atlas prices **perishable assets** — contracts whose value resolves from a
hard settlement deadline — against a professional counterparty whose quotes
move every few seconds, on a venue with no public API: the market exists only
inside a browser page. The engine behaves the way a market-maker reads a book:
derive the fair value from first principles, compare it to the quote, and
refuse the trade almost all of the time.

Everything about Atlas is built around a refusal. It refuses to enter anything
it cannot price. It refuses a stake it cannot justify. It refuses certainty it
cannot earn. When the evidence says the edge is not real, it refuses to keep
playing — automatically, and before the money moves.

> The most dangerous number in this system is a 1.0.
> No live outcome is ever certain. Trust no model that claims otherwise.

---

## What it is · What it is not

**Atlas is —**

- A closed-loop decision engine: capture → posterior → gate → size → execute
  → settle → recalibrate, running unattended on real money.
- An engineering study in honest probability: isotonic calibration learned
  from its own settled ledger, fractional Kelly sizing, pre-registered success
  criteria, and a verdict mechanism that can shut the whole operation down.
- An infrastructure deployment project with operator discipline: single-shot
  execution path, watchdogged sessions, read-only observatory, and an
  append-only evidence ledger.

**Atlas is not —**

- A prediction machine. It claims no crystal ball. The pricing core is
  deliberately **deterministic** — no neural networks, no black-box "AI picks".
- A profit rig with published performance claims. Its edge must be proven on
  its own ledger, at a pre-registered sample size, before flow is permitted —
  and it is designed to freeze itself if that proof fails.
- A shadow scraper. It drives a real logged-in session, in the open, with
  every action logged.

---

## Architecture

One process, two planes. The fast plane is pure mathematics. The slow plane is
control: sessions, gates, ledgers, verdicts.

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
                       ║     │    state space (Poisson / Dixon–Coles)            │
                       ║     │                           │                      │
                       ║     │                           ▼                      │
                       ║     │    every contract priced off ONE joint matrix     │
                       ║     │    (marginals · combinators · path contracts)     │
                       ║     │                           │                      │
                       ║     │                           ▼                      │
                       ║     │    isotonic (PAVA) calibration · per family       │
                       ║     │                           │                      │
                       ║     │                           ▼                      │
                       ║     │    fractional Kelly sizing, hard floor and cap    │
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
                             exception recovery ONLY · zero execution
                             tools · zero pricing authority

  deployed as systemd units:  runner (loop) · dashboard · browser session
                              watchdog (oneshot) · settlement timer
```

The separation is absolute and deliberate:

- **The mathematical engine is the only source of price.** No LLM ever sees,
  adjusts, or overrides a probability.
- **The LLM never executes.** It exists for the one job automation is worst
  at — recovering from unpredictable session failures — and holds no
  execution authority of any kind.
- **The execution path is single-shot.** `click`/`fill`/`upload` run once.
  Ambiguity is resolved by evidence, never by retry. A second click can be a
  second position.

---

## Pricing: one posterior, every contract

The venue gives you a clock, a scoreboard of state variables, and quotes. The
engine derives fair value from first principles, in real time:

1. **Remove the venue's margin.** Three independent models — proportional,
   power, and Shin — strip the quoted overround out of the offered prices,
   yielding a margin-free implied distribution. Where the three disagree, the
   disagreement is treated as information, not noise.
2. **Estimate elapsed time honestly.** A hazard-aware model converts the
   visible clock into effective remaining time, correcting for the deadline
   inflation the venue displays to you.
3. **Infer the latent process parameters.** Ex-ante rates are recovered from
   quoted prices; live state variables (patterns of pressure, discipline, and
   tempo visible in the state feed) update them in a posterior, with
   empirical-Bayes shrinkage on thin inputs.
4. **Build one joint posterior.** A Poisson/Dixon–Coles matrix (with the τ
   correlation correction) over the remaining state space becomes the single
   world model.
5. **Price every contract off that one matrix.** Linears, totals, spreads,
   props, path-and-elimination-shaped contracts and their combinations are all
   marginal/combinatorial reads of the same joint distribution — no ad-hoc
   per-contract models, therefore no possibility of internal contradiction
   between two quotes the engine prices differently.
6. **Compute EV, honestly.** Every decision carries `model_p`, `calibrated_p`,
   `ev`, and `ev_lower`. Acceptance uses the *low* estimate, not the
   flattering one.

A wrong price is survivable. Two contracts priced by two different models that
disagree with each other is not — that is one position hidden in another, and
Atlas's single-matrix rule makes it structurally impossible.

---

## Why calibration exists (the honest part)

An early live window produced the numbers that motivated this layer. Across
that window the ledger's claimed edge averaged +13.1% while realised return
came in near +9.5% — a disappointment, within noise. The indictment was more
specific: one perishable, hardest-to-price family had priced roughly +8.9% of
claimed EV and returned −38.3% realised. Small sample, one family, one window
— which is exactly why the response is statistical, not anecdotal.

Atlas fixes this without deleting history. It re-linearises it: an **isotonic
(PAVA) calibration map is fit per contract family from settled outcomes**,
mapping raw `model_p` to an empirically-corrected `calibrated_p` — and it is
the calibrated number, not the raw one, that gates EV. Families without at
least 50 settled observations are *not* calibrated; they trade on
shrinkage-adjusted raw probabilities and say so, loudly, in the decision
record.

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
probability and the observed quote, bounded by a floor, a per-position cap, a
per-object exposure cap, and the campaign flat unit when the operator is
silent.

Kelly on a miscalibrated p is a money shredder. Kelly on a *calibrated* p,
fractionally discounted, with hard floors and caps — that is a defensible
algorithm.

---

## The risk gates

Every candidate must pass every gate. The refusal reasons are recorded
word-for-word, so the ledger doubles as a forensic record of what the system
thought and why it said no.

| Gate | Threshold | Meaning of the line |
|---|---|---|
| Minimum EV | ≥ 0.05 | Below 5% edge, quote noise will eat it — no trade. |
| EV ceiling | ≤ 0.25 | An EV too good to be true usually is. Something is wrong upstream; do not touch it. |
| Certainty ceiling | p < 0.95 | A model claiming ≥ 95% for any live outcome is broken — abstain. p = 1.0 is always refused. |
| Quote ceiling | ≤ 3.0 | At long odds a 1% probability error is the whole stake. The 100-unit loss at 10.0× was the case study. |
| Probe filter | — | Inputs that smell like test/debug artefacts are refused outright. |
| Duplicate cap | 1 per selection | One position per selection. No averaging down. No re-entry. |
| Exposure cap | per-object & global | Correlated positions cannot stack beyond the cap. |
| Family stop-loss | 3 consecutive settled losses | A contract family that keeps losing freezes itself until a win resets the streak. |
| Campaign wall | `proven` only | **Execution is fail-closed.** Until the campaign record says "proven", no order leaves — collect / watch / observe / missing state / scorer errors all freeze before the slip. |

The campaign is pre-registered. Success criteria were fixed *before* capital
was deployed, and the verdict is recomputed from the ledger every hour:

| Verdict | Criterion | Consequence |
|---|---|---|
| `collect` | fewer than 100 settled positions | observe only · no execution |
| `watch` / `observe` | sample growing, proof not yet hard | observe only · no execution |
| `proven` | ROI confidence interval excludes zero **and** CLV ≥ 0 | gates open, within all caps |
| anything else | missing state, stale scorer data, scoring error | **fail-closed** · no execution |

> The pre-registration is the point. Atlas decided what success would look like
> *before* deploying capital, then lets the settled outcomes speak — and binds
> the machine to whatever they say. A freeze is not a failure of the system;
> it is the system working.

---

## Execution plumbing: production infrastructure discipline

This is not a notebook that gets run on enthusiasm. It is a deployed service
stack with an opinion about every failure mode:

| Component | Discipline |
|---|---|
| systemd unit topology | `atlas-loop` (gated by an arm flag — no flag, no loop), `atlas-dashboard`, `atlas-browser`, `atlas-watchdog` (oneshot, timer-driven), `atlas-settle.timer` (hourly reconciliation). |
| Capture hot path | One evaluate-injection bundle per cycle; the whole board parses in ~a dozen seconds, not minutes. Read timeouts of 5 s so a wedged page fails fast instead of stalling the loop. |
| Execution path | Single-shot, no retries, one position per selection. |
| Session liveness | Heartbeat freshness gates everything; a stale heartbeat flips the dashboard to "not submitting" — by construction, not by monitoring alarm. |
| Wedge recovery | A oneshot watchdog timer kills and recovers wedged browser/daemon processes before an operator would notice. |
| Posture steering | Risk posture (EV floor, Kelly fraction, caps) is configuration — the dashboard can tighten it, never loosen it. The asymmetric-ratchet rule: anything that can only make the system more conservative is allowed; nothing else is. |
| Reconciliation | A settlement timer reconciles the venue's own history against the ledger hourly; missing settlements are surfaced, never assumed. |

No headless masquerade, no shadow scraping: Atlas drives the real logged-in
session — the same account you would drive yourself — at a pace you can watch,
with every action logged.

---

## Where the AI is allowed — and is not

Atlas is an AI system that decided where AI doesn't belong.

- **No neural networks in pricing.** A live price needs a defensible
  deterministic derivation at cycle speed: de-vig models, Poisson/Dixon–Coles
  posteriors, isotonic maps — not opaque learned approximations that drift
  silently under regime change.
- **The LLM has no authority over money.** It recovers a wedged session by
  re-authenticating it. It never proposes, modifies, or approves a position.
  It does not see the execution path. The wall is architectural, not advisory.
- **No LLM-generated telemetry.** Every number in the ledger comes from the
  ledger itself — from the observable state, the clock, the quotes, or the
  settlement.

---

## Evidence: the ledger is the product

Everything decision-shaped is recorded on append-only JSONL:

- `bet_placed` — what, when, at what quote, at what calibrated probability, at
  what stake, at what gate state
- `bet_settled` — outcome, P&L, the realised-vs-claimed delta
- `decision` & `gate refusal` records — the reasoning at each gate, including
  all failures

The campaign scorer reads the ledger and prints the verdict. The operations
dashboard (`http://…/status`) is a **read-only observer**: heartbeat liveness,
exposure split (recent / unreconciled), posture state, and the one thing that
matters — the campaign gate state. It can tighten posture, but it cannot force
an order.

Records are versioned, checksummed, and retained. OTLP telemetry exists, but
is optional — evidence is designed to survive without any observability stack.

---

## Quickstart

```bash
git clone https://github.com/EmmanuelAdesina/atlas.git
cd atlas
pip install -e ".[test]"

# Full regression suite — includes the case studies below as tests
python3 -m pytest -v

# Replay a recorded live observation through the full pricing engine
python3 -m atlas.main live-demo \
  --fixture tests/fixtures/fortuna_stuttgart_90plus.json \
  --log data/decisions/demo.jsonl

# Post-settlement audit + calibration forensics from a decision log
python3 -m atlas.main boss \
  --fixture tests/fixtures/fortuna_stuttgart_90plus.json \
  --log data/decisions/boss_audit.jsonl

# Read-only operations dashboard (posture, gates, campaign state)
python3 -m atlas.main run --dashboard --port 8077
```

Deployment notes:

- Venue credentials live only in environment variables — never in code (see
  `atlas/config/settings.py`).
- A WebBridge daemon must be reachable at `ATLAS_WEBBRIDGE_DAEMON_URL`.
- The exception-recovery supervisor takes optional LLM credentials
  (`ATLAS_BOSS_BASE_URL`, `ATLAS_BOSS_MODEL`, `ATLAS_BOSS_API_KEY`).

**Nothing executes until the arm flag exists and submission is explicitly
enabled.** The default posture is observe-and-log; real money additionally
requires a proven campaign record and every gate green — the campaign wall
enforces this at the last step before execution, and it fails closed.

---

## Case studies: failures taught us the gates

These are real incidents, in order, and the engineering responses. None are
hypothetical; all are grounded in the gate set you can read in the ledger.

1. **The certainty catastrophe.** The engine claimed p = 1.0 on contracts
   quoted at 10.0× and 13.5×. The stake was defensible only if the claim was
   true — it was not. **Response:** a hard probability ceiling (`p < 0.95`
   refusal) and a certainty gate that treats any 1.0 as an emergency, not an
   entry.
2. **Long-odds poisoning.** Flat 100-unit stakes landed on lottery-shaped
   quotes — 57.0×, 10.0× — where the entire edge claim collapses if the
   probability model is wrong by a single percentage point. It was.
   **Response:** a decimal-odds ceiling. Calibration can correct the *centre*
   of a probability estimate; it cannot defend a stake at odds where the error
   bars are the size of the decision.
3. **The repeat-entry loop.** The same selection was re-entered after the
   first attempt had already gone against it. **Response:** single-shot
   execution path with no retries, one-position-per-selection cap, and the
   fail-closed campaign wall.
4. **The calibration gap.** While a family was uncalibrated, it traded on raw
   model probabilities that had never faced a settlement sample. **Response:**
   families with fewer than 50 settled observations are flagged as
   uncalibrated — they trade on shrinkage-adjusted raw probabilities, and
   every decision record says so explicitly.

Each gate exists because a real loss earned it. The refusal log quotes its own
gate parameters — every *why* a trade didn't happen is in the ledger forever.

---

## Repository

```
atlas/
├ atlas/live/           the two planes: math_engine (pricing), capture,
│                       estimate, decisions (gates), calibration, runner,
│                       control (dashboard), supervisor, boss
├ atlas/ops/            ledger, campaign verdict, posture, portfolio, settle
│                       + settle_history (reconciliation), telemetry, notify
├ atlas/browser/        WebBridge adapter: slip, slipclear, cashout actions
├ atlas/collectors/     venue adapters: board collection, cross-venue reference
├ atlas/mappers/        venue → canonical → venue
├ atlas/models/         canonical domain models
├ atlas/config/         settings, environment
├ atlas/http/           retry-safe HTTP client
├ tests/                the fixtures, the case studies, the guarantees
├ docs/                 mission · engineering · known issues (kept honest)
└ data/                 runtime state (gitignored): ledger, calibration maps,
                        campaign verdict, heartbeats
```

---

## Design invariants

1. **Execution is OFF until it is explicitly ON.** No code path enables real
   flow silently.
2. **The campaign wall is fail-closed.** Missing state = no money moved.
3. **One selection, one position.** Forever.
4. **p = 1.0 is a bug report, not a trade.**
5. **Never enters what it cannot price.** Unknown markets are refused, not
   guessed.
6. **The execution path does not retry.** Ever. Evidence replaces attempts.
7. **The LLM recovers sessions, never prices markets, never moves money.**
8. **The ledger survives everything.** Append-only JSONL; the settlement is
   the scoreboard of truth.
9. **"Unknown" is a valid result.** The system returns "unattributable"
   instead of manufacturing causal stories.
10. **Negative results are results.** A frozen campaign is a working outcome.

> The question Atlas was built around: *when the system's assumptions are
> wrong, what does it do?* Every line above is one answer to that question.

---

## Documentation

- [`docs/mission.md`](docs/mission.md) — the objective and non-negotiables
- [`docs/engineering.md`](docs/engineering.md) — architecture decisions and
  mathematical invariants
- [`docs/bugs_and_issues.md`](docs/bugs_and_issues.md) — the honest, open list

---

## Disclosure & responsibility

- Atlas operates a research account with small, capped, self-limiting stakes
  in a live real-money venue. It is not financial advice, not a prediction
  service, and not a product for resale.
- The entire construction assumes the counterparty is usually right, and
  demands pre-registered evidence before it is allowed to be wrong.
- Use responsibly, within local regulation. When in doubt, the system is
  designed — deliberately — to do the boring thing: nothing.

---

<div align="center">

### Atlas

*Built by an engineer who trusts gates more than opinions, ledgers more than
claims, and calibration more than convictions.*

</div>
