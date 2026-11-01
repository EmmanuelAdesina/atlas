"""OpenTelemetry wiring for Atlas: metrics + traces, pushed OTLP/HTTP.

Atlas no longer has an HTTP dashboard, and that is deliberate: the trading
box must expose ZERO listening sockets. The only surface it offers is
outbound OTLP push to a collector (ATLAS_OTLP_ENDPOINT). Anyone who wants
loop state pulls the exported records from the collector — never from the
trading box itself.

The invariant this module protects: telemetry failure disables monitoring
QUALITY, not the loop. Every opentelemetry import is guarded; if the SDK or
the exporter is not installed, Telemetry degrades to a no-op facade and the
loop keeps running. Every Telemetry method also swallows exceptions (logged
only) so a broken exporter can never take a cycle down with it — same
contract as Notifier: a broken sink must not break the loop.

Config, read by init_telemetry():
  - ATLAS_OTLP_ENDPOINT — OTLP/HTTP (protobuf) base, e.g. http://collector:4318.
    /v1/metrics and /v1/traces are appended when not already present.
    Unset -> telemetry disabled (one setup line, then full no-op).
  - ATLAS_OTLP_HEADERS  — optional "k=v;k=v" exporter headers (auth).
  - OTEL_SERVICE_NAME   — service.name resource attribute; defaults to
    "atlas-live-loop", with the session name appended when given.
"""

from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from typing import Any, Callable, Iterator, Optional

logger = logging.getLogger(__name__)

# --- guarded opentelemetry imports ------------------------------------------
# If anything OT-shaped is missing, the module still imports (stdlib only)
# and _OTEL_AVAILABLE stays False. NO fastapi/uvicorn imports here — this
# module must load in a bare interpreter as a pure no-op.
_OTEL_AVAILABLE = False
try:
    from opentelemetry import metrics as _otel_metrics
    from opentelemetry import trace as _otel_trace
    import opentelemetry.exporter.otlp.proto.http.metric_exporter as _me_mod
    import opentelemetry.exporter.otlp.proto.http.trace_exporter as _te_mod
    from opentelemetry.sdk.metrics import MeterProvider
    from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
    from opentelemetry.sdk.metrics.view import (
        ExplicitBucketHistogramAggregation,
        View,
    )
    from opentelemetry.sdk.resources import Resource
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import BatchSpanProcessor

    # Exporter class names drifted across OTel releases (the HTTP trace
    # exporter is OTLPSpanExporter in >=1.30, OTLPTraceExporter in older
    # trees). Resolve either; a release that renames both disables telemetry
    # below rather than crashing the import.
    OTLPMetricExporter = getattr(_me_mod, "OTLPMetricExporter", None)
    OTLPTraceExporter = getattr(_te_mod, "OTLPTraceExporter", None) or getattr(
        _te_mod, "OTLPSpanExporter", None
    )
    if OTLPMetricExporter is None or OTLPTraceExporter is None:
        raise ImportError("otlp proto http exporter classes not found")

    _OTEL_AVAILABLE = True
except ImportError:  # pragma: no cover - exercised only without OTel installed
    MeterProvider = None  # type: ignore[assignment]


# Gauge names. The five observable gauges are a fixed vocabulary over one
# meter; the CALLBACKS behind them are supplied by the caller at runtime
# via set_gauge_source (e.g. the loop wiring report.ledger.daily_pnl in).
_GAUGE_NAMES = (
    "atlas.heartbeat_age_s",
    "atlas.daily_pnl_ngn",
    "atlas.open_exposure_ngn",
    "atlas.posture_max_stake_ngn",
    "atlas.stop_active",
)


# --- no-op fallbacks (shape-compatible, cost nothing) ------------------------


class _NoOpInstrument:
    """Accepts add()/record()/set() calls and drops them on the floor."""

    def add(self, amount: float, attributes: Optional[dict] = None) -> None:
        pass

    def record(self, amount: float, attributes: Optional[dict] = None) -> None:
        pass


@contextmanager
def _no_op_span(name: str, attrs: Optional[dict] = None) -> Iterator[None]:
    yield None


def _is_noop(obj: Any) -> bool:
    return isinstance(obj, _NoOpInstrument)


# --- endpoint helpers --------------------------------------------------------


def _split_endpoint(base: str) -> tuple[str, str]:
    """Split a base endpoint into (metrics_url, traces_url).

    OTLP/HTTP lands on <base>/v1/metrics and <base>/v1/traces (protobuf).
    If the caller already gave a full /v1 path for one signal, the other is
    derived from the same collector prefix — a misconfigured half only costs
    half the telemetry, not the loop.
    """
    b = base.rstrip("/")
    if b.endswith("/v1/metrics"):
        prefix = b[: -len("/v1/metrics")]
    elif b.endswith("/v1/traces"):
        prefix = b[: -len("/v1/traces")]
    else:
        prefix = b
    return prefix + "/v1/metrics", prefix + "/v1/traces"


def _parse_headers(raw: str) -> dict[str, str]:
    """'k=v;k=v' -> {'k': 'v'}. Malformed pairs are dropped, not fatal."""
    headers: dict[str, str] = {}
    for pair in raw.split(";"):
        pair = pair.strip()
        if not pair or "=" not in pair:
            continue
        key, _, value = pair.partition("=")
        key = key.strip().strip('"').strip("'")
        value = value.strip().strip('"').strip("'")
        if key:
            headers[key] = value
    return headers


def _sanitize_session(session: str) -> str:
    return "".join(c if c.isalnum() or c in "-_" else "-" for c in session).strip("-")


def _console_note(message: str) -> None:
    """One human-visible setup line. rich if present, logging as fallback."""
    try:
        from rich.console import Console  # local import; rich is optional

        Console().print(f"[cyan]{message}[/cyan]")
        return
    except Exception:
        pass
    logger.info(message)


# --- the facade --------------------------------------------------------------


class Telemetry:
    """Metrics + traces for the live loop, all off one meter/tracer.

    How it fails when it's misconfigured: it doesn't crash. If ATLAS_OTLP_ENDPOINT
    is unset, or opentelemetry is not installed, or the exporter can't reach the
    collector, every method becomes a no-op (or logs and returns) and the loop
    runs blind. Telemetry failure = lost monitoring, never a lost cycle.
    """

    def __init__(
        self,
        session: str = "",
        endpoint: Optional[str] = None,
        enabled: bool = _OTEL_AVAILABLE,
    ) -> None:
        self.session = session
        self.endpoint = (endpoint or os.environ.get("ATLAS_OTLP_ENDPOINT", "") or "").strip()
        self.enabled = bool(enabled)
        self._meter: Any = None
        self._tracer: Any = None
        self._meter_provider: Any = None
        self._tracer_provider: Any = None

        # Caller-registered gauge sources: name -> callable() -> float.
        # Gauges fire only from registered sources; probing an unregistered
        # gauge yields no datapoint rather than a fake 0.0.
        self._gauge_sources: dict[str, Callable[[], float]] = {}

        # Counters (built per-instrument in _build_instruments).
        self.cycles_total: Any = _NoOpInstrument()
        self.duration_s_hist: Any = _NoOpInstrument()
        self.opportunities_total: Any = _NoOpInstrument()
        self.planned_total: Any = _NoOpInstrument()
        self.placed_total: Any = _NoOpInstrument()
        self.settled_total: Any = _NoOpInstrument()
        self.arb_signals_total: Any = _NoOpInstrument()
        self.cashout_taken_total: Any = _NoOpInstrument()

        if not self.endpoint:
            self.enabled = False
        if not self.enabled:
            if _OTEL_AVAILABLE and not self.endpoint:
                _console_note(
                    "telemetry disabled: ATLAS_OTLP_ENDPOINT not set "
                    "(no push surface configured; loop continues blind)"
                )
            elif not _OTEL_AVAILABLE:
                _console_note(
                    "telemetry disabled: opentelemetry/otlp-exporter not installed "
                    "(monitoring quality lost, loop unaffected)"
                )
            return

        try:
            self._build()
        except Exception:
            # An SDK wiring problem (bad endpoint, missing exporter dep,
            # malformed resource) degrades telemetry to no-op — never crashes.
            logger.exception("telemetry init failed; running as no-op")
            self.enabled = False
            self._meter_provider = None
            self._tracer_provider = None

    # --- construction ------------------------------------------------------

    def _service_name(self) -> str:
        name = os.environ.get("OTEL_SERVICE_NAME", "") or "atlas-live-loop"
        if self.session:
            name = f"{name}-{_sanitize_session(self.session)}"
        return name or "atlas-live-loop"

    def _build(self) -> None:
        metrics_url, traces_url = _split_endpoint(self.endpoint)
        headers = _parse_headers(os.environ.get("ATLAS_OTLP_HEADERS", "") or "")
        resource = Resource.create(
            {
                "service.name": self._service_name(),
                "service.namespace": "atlas",
            }
        )
        timeout_s = float(os.environ.get("ATLAS_OTLP_TIMEOUT_S", "10") or 10)

        # Metrics: one PeriodicExportingMetricReader pushing protobuf OTLP.
        # Content type stays the exporter default ("application/x-protobuf");
        # we never override it — the protobuf default is the contract with the
        # collector, same encoding the loop's signals ride.
        metric_exporter = OTLPMetricExporter(
            endpoint=metrics_url,
            headers=headers or None,
            timeout=timeout_s,
        )
        # Duration boundaries are seconds-scale: the loop cycles in tens of
        # seconds; boundaries below 0.1s would bucket nothing but overheads.
        provider_kwargs: dict[str, Any] = {
            "resource": resource,
            "views": [
                View(
                    instrument_name="atlas.cycle.duration_s",
                    aggregation=ExplicitBucketHistogramAggregation(
                        boundaries=[1.0, 5.0, 15.0, 30.0, 60.0, 120.0, 300.0, 600.0, 1800.0, 3600.0],
                    ),
                )
            ],
        }
        self._meter_provider = MeterProvider(
            metric_readers=[
                PeriodicExportingMetricReader(
                    metric_exporter,
                    export_interval_millis=int(float(os.environ.get("ATLAS_OTLP_INTERVAL_S", "15") or 15) * 1000),
                )
            ],
            **provider_kwargs,
        )

        # Observable gauges: registered once here, backed live by whatever
        # sources the caller has registered at collection time. Gauges only
        # emit for names whose source is registered — the closure below
        # enumerates self._gauge_sources rather than hardcoding all five.
        self._meter = self._meter_provider.get_meter("atlas.live")

        def _gauge_callback(gauge_name: str) -> Callable[[Any], list[Any]]:
            def callback(options: Any) -> list[Any]:
                source = self._gauge_sources.get(gauge_name)
                if source is None:
                    return []
                try:
                    value = float(source())
                except Exception:
                    # A misbehaving source yields no observation; it must not
                    # poison the provider's collection thread.
                    return []
                observation_cls = getattr(_otel_metrics, "Observation", None)
                if observation_cls is not None:
                    return [observation_cls(value)]
                return [value]

            return callback

        for gauge_name in _GAUGE_NAMES:
            try:
                self._meter.observe_gauge(
                    gauge_name,
                    callbacks=[_gauge_callback(gauge_name)],
                )
            except Exception:
                # Older SDK spelling: create_observable_gauge.
                try:
                    self._meter.create_observable_gauge(
                        gauge_name,
                        callbacks=[_gauge_callback(gauge_name)],
                    )
                except Exception:
                    logger.exception("gauge %s not registered; skipping", gauge_name)

        # Traces: BatchSpanProcessor so span export never happens mid-cycle.
        self._tracer_provider = TracerProvider(resource=resource)
        self._tracer_provider.add_span_processor(
            BatchSpanProcessor(
                OTLPTraceExporter(
                    endpoint=traces_url,
                    headers=headers or None,
                    timeout=timeout_s,
                )
            )
        )
        self._tracer = self._tracer_provider.get_tracer("atlas.live")

        self._build_instruments()

    def _build_instruments(self) -> None:
        """Create every counting instrument up front. All attribute sets stay
        low-cardinality: outcome is a 5-value enum, never a bet/match id."""
        meter = self._meter

        def _mk(instrument_name: str, description: str) -> Any:
            try:
                return meter.create_counter(instrument_name, description=description)
            except Exception:
                logger.exception("counter %s not created; no-op", instrument_name)
                return _NoOpInstrument()

        def _mk_hist(instrument_name: str, description: str) -> Any:
            try:
                return meter.create_histogram(instrument_name, description=description)
            except Exception:
                logger.exception("histogram %s not created; no-op", instrument_name)
                return _NoOpInstrument()

        self.cycles_total = _mk(
            "atlas.cycles_total",
            "Loop cycles by outcome (ok|not_ready|crashed|finished|stop_active). "
            "Proves the loop LIVED; says nothing about correctness of a cycle.",
        )
        self.duration_s_hist = _mk_hist(
            "atlas.cycle.duration_s",
            "Cycle wall duration in seconds. Proves how long the loop took; "
            "not what it did inside.",
        )
        self.opportunities_total = _mk(
            "atlas.opportunities_total",
            "Opportunities surfaced by scanning. Proves scan yield; zero while "
            "not_ready is normal, not a scan outage signal by itself.",
        )
        self.planned_total = _mk(
            "atlas.planned_total",
            "Bets planned by sizing. Proves the gate produced plans.",
        )
        self.placed_total = _mk(
            "atlas.placed_total",
            "Bets actually placed. Proves reach of the book, not quality.",
        )
        self.settled_total = _mk(
            "atlas.settled_total",
            "Bets settled (won/lost/void). Proves reconciliation ran.",
        )
        self.arb_signals_total = _mk(
            "atlas.arb_signals_total",
            "Sure-thing arb signals seen. Proves the detector fired.",
        )
        self.cashout_taken_total = _mk(
            "atlas.cashout_taken_total",
            "Cashouts executed. Proves the exit path was exercised.",
        )

    # --- gauge plumbing ------------------------------------------------------

    def set_gauge_source(self, name: str, source: Callable[[], float]) -> None:
        """Register the callback backing observable gauge `name`.

        The gauge instrument already exists (built at init); this only feeds
        its callback. Unknown gauge names are logged and ignored — a typo in
        the source registration must not grow the metric vocabulary.
        """
        if name not in _GAUGE_NAMES:
            logger.warning("set_gauge_source: unknown gauge %r; ignored", name)
            return
        self._gauge_sources[name] = source

    # --- metrics -------------------------------------------------------------

    def note_cycle(
        self,
        outcome: str,
        ready: bool,
        opportunities: int = 0,
        planned: int = 0,
        placed: int = 0,
        settled: int = 0,
        arbs: int = 0,
        cashouts: int = 0,
        duration_s: float = 0.0,
        daily_pnl_ngn: float = 0.0,
        stop_active: bool = False,
        **attrs: Any,
    ) -> None:
        """Report a COMPLETED cycle (after the fact: duration is measured by
        the caller, not by us) and emit the matching atlas.cycle span.

        Counter attributes stay low-cardinality — outcome only, from
        {"ok", "not_ready", "crashed", "finished", "stop_active"}. Everything
        else rides as span attributes, where cardinality is cheap.
        Never raises.
        """
        try:
            if outcome not in ("ok", "not_ready", "crashed", "finished", "stop_active"):
                outcome = "crashed" if outcome == "" else outcome
            counter_attrs = {"outcome": outcome}
            self.cycles_total.add(1, attributes=counter_attrs)
            self.duration_s_hist.record(max(0.0, float(duration_s)), attributes=counter_attrs)
            if opportunities:
                self.opportunities_total.add(opportunities, attributes=counter_attrs)
            if planned:
                self.planned_total.add(planned, attributes=counter_attrs)
            if placed:
                self.placed_total.add(placed, attributes=counter_attrs)
            if settled:
                self.settled_total.add(settled, attributes=counter_attrs)
            if arbs:
                self.arb_signals_total.add(arbs, attributes=counter_attrs)
            if cashouts:
                self.cashout_taken_total.add(cashouts, attributes=counter_attrs)
        except Exception:
            logger.exception("note_cycle: metric recording failed; continuing")

        try:
            span_attrs: dict[str, Any] = {
                "atlas.cycle.outcome": outcome,
                "atlas.cycle.ready": bool(ready),
                "atlas.cycle.opportunities": int(opportunities),
                "atlas.cycle.planned": int(planned),
                "atlas.cycle.placed": int(placed),
                "atlas.cycle.settled": int(settled),
                "atlas.cycle.arbs": int(arbs),
                "atlas.cycle.cashouts": int(cashouts),
                "atlas.cycle.duration_s": float(duration_s),
                "atlas.daily_pnl_ngn": float(daily_pnl_ngn),
                "atlas.stop_active": bool(stop_active),
            }
            span_attrs.update(attrs)
            if self._tracer is not None and not _is_noop(self._tracer):
                # Report-after-the-fact span: created and closed immediately,
                # carrying the cycle's numbers. No children.
                span = self._tracer.start_span("atlas.cycle", attributes=span_attrs)
                span.end()
        except Exception:
            logger.exception("note_cycle: span emission failed; continuing")

    # --- spans ----------------------------------------------------------------

    @contextmanager
    def span(self, name: str, attrs: Optional[dict] = None) -> Iterator[Any]:
        """Span context for placement/settle flows. Errors are recorded on
        the span (OTel default) and re-raised — this observes, never swallows.
        """
        if self.enabled and self._tracer is not None and not _is_noop(self._tracer):
            cm = self._tracer.start_as_current_span(name, attributes=dict(attrs or {}))
            with cm as s:
                yield s
        else:
            yield None

    # --- lifecycle --------------------------------------------------------------

    def shutdown(self) -> None:
        """Flush and stop the providers. Best-effort in both directions:
        never raises, and safe to call more than once (providers tolerate a
        second shutdown; if not, the guard below catches it)."""
        for attr in ("_meter_provider", "_tracer_provider"):
            provider = getattr(self, attr, None)
            if provider is None:
                continue
            try:
                provider.shutdown()
            except Exception:
                logger.exception("telemetry shutdown: %s.flush failed", attr)
            finally:
                setattr(self, attr, None)
        self.enabled = False


# --- module-level entry point ---------------------------------------------------

_current: Optional[Telemetry] = None
_init_lock = threading.Lock()


def init_telemetry(session: str = "") -> Telemetry:
    """Idempotent setup of the loop's telemetry. Returns the SAME Telemetry
    on every call — re-initializing would re-register gauges and spawn a
    second exporter, so the first one wins for the process lifetime."""
    global _current
    with _init_lock:
        if _current is None:
            _current = Telemetry(session=session)
        return _current
