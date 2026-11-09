"""Tracing with Arize Phoenix — see every step the agent takes, and every LLM call inside.

The ``steps`` list in the API response says *which* nodes ran. A trace says much more:
for each node, the exact prompt sent to the model, the raw reply, the parsed judgment,
the retrieved chunks with their scores, token counts and latency — nested as a tree
(request -> graph -> node -> chain -> LLM call). It is how you debug an agent: "why did
the grader reject this chunk?" becomes a click instead of a guess.

How it works:
  - **OpenTelemetry** is the vendor-neutral standard for traces: the app produces
    *spans* (timed, nested units of work) and an *exporter* sends them to a backend;
  - **OpenInference** (``LangChainInstrumentor``) hooks into LangChain's callback system,
    which LangGraph uses too, and turns every Runnable run into a span — no change to
    the agent's code;
  - **Phoenix** is the backend: a local server (http://localhost:6006) that stores the
    spans and shows them. Everything stays on the machine, in line with the project.

Tracing is off by default (``COPILOT_TRACING_ENABLED``): the app must not depend on a
monitoring server being up. The tracing packages are imported only when it is enabled.
"""

import logging

from copilot.config import settings

logger = logging.getLogger(__name__)

_tracer_provider = None


def setup_tracing(span_exporter=None):
    """Instrument LangChain/LangGraph and export spans. Safe to call more than once.

    By default spans go to the Phoenix server in ``settings.phoenix_endpoint``. Passing a
    ``span_exporter`` sends them there instead (the tests use an in-memory exporter, so
    they can inspect the spans without a running server).

    Returns the tracer provider, or ``None`` when tracing is disabled.
    """
    global _tracer_provider
    if _tracer_provider is not None:
        return _tracer_provider
    if not settings.tracing_enabled and span_exporter is None:
        logger.info("Tracing disabled (set COPILOT_TRACING_ENABLED=true to enable it).")
        return None

    from openinference.instrumentation.langchain import LangChainInstrumentor
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from phoenix.otel import register

    if span_exporter is None:
        # batch=True: spans are sent in the background, so a slow or stopped Phoenix
        # server never slows the agent down (failed exports are only logged).
        provider = register(
            endpoint=settings.phoenix_endpoint,
            project_name=settings.phoenix_project,
            batch=True,
            verbose=False,
        )
        logger.info("Tracing enabled: spans sent to %s", settings.phoenix_endpoint)
    else:
        # A plain OpenTelemetry provider: no network, spans go only to the given exporter.
        provider = TracerProvider()
        provider.add_span_processor(SimpleSpanProcessor(span_exporter))

    LangChainInstrumentor().instrument(tracer_provider=provider)
    _tracer_provider = provider
    return provider
