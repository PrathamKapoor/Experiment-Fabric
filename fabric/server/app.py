"""The HTTP API, on the standard library.

No web framework, for the same reason the rest of the core has no
dependencies: a researcher should be able to run this on a cluster login
node with whatever Python is already there.  ``http.server`` is enough
for an API of this shape, and its limits are stated rather than hidden
(see :class:`FabricServer`).

Every handler is a thin translation.  It parses the request, calls one
function in :mod:`fabric.server.service`, and serialises the result.  No
handler contains logic the command line does not also reach, which is the
rule section 31 sets and the only thing keeping the two from drifting.
"""

from __future__ import annotations

import json
import os
import re
import socket
import threading
import time
import traceback
import urllib.parse
from collections.abc import Callable
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from fabric import __version__
from fabric.core.store import Workspace
from fabric.server import service
from fabric.server.jobs import JobStore
from fabric.server.worker import WorkerPool
from fabric.util.jsonio import dumps as json_dumps

API_VERSION = "1.0"

#: Largest request body accepted, in bytes. A specification is a few
#: kilobytes; anything approaching a megabyte is a mistake or an attack,
#: and reading it into memory first would be both.
MAX_BODY_BYTES = 1 << 20

_CSRF_REFUSAL: dict[str, object] = {
    "error": "refused: this looks like a cross-origin request",
    "status": 403,
    "remedy": "Fabric's API refuses state-changing requests that a web page "
              "initiated, because a page cannot prove it is talking to this "
              "server on purpose. Open the Fabric interface itself and issue "
              "the request there, or use the fabric command line, which does "
              "not go through HTTP.",
}

#: How much of an over-long body to read and throw away before answering.
#:
#: Refusing without reading is the obvious implementation and the wrong
#: one: the client is still writing, so it gets a connection reset instead
#: of the 413 explaining what happened. Draining a bounded amount first
#: means the caller sees a real answer. Past this limit there is nothing
#: to be gained by continuing to read, and the connection is dropped.
DRAIN_LIMIT_BYTES = 16 * MAX_BODY_BYTES

#: Where the single-page interface lives on disk.
UI_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui")


@dataclass
class Route:
    """One method-and-path pattern, and what handles it."""

    method: str
    pattern: re.Pattern[str]
    handler: Callable[..., Any]
    description: str = ""


class FabricServer:
    """The application: a workspace, a job queue, workers, and routes.

    Deliberately small. ``ThreadingHTTPServer`` is not a production web
    server -- no HTTP/2, no graceful load shedding, no TLS -- and Fabric
    does not pretend otherwise: put a real reverse proxy in front of it if
    it is exposed to anything but a trusted network. What it *is* is a
    dependency-free way to run the same domain objects the command line
    uses over HTTP, which is what makes the deployment path real instead
    of aspirational.
    """

    def __init__(self, workspace: Workspace, *, execute: bool = False,
                 workers: int = 1, database: str = "",
                 serve_ui: bool = True) -> None:
        self.workspace = workspace
        self.execute = execute
        self.started_at = time.time()
        self.serve_ui = serve_ui
        self.jobs = JobStore(database or os.path.join(workspace.path, "jobs.sqlite3"))
        self.pool = WorkerPool(self.jobs, workspace, size=workers, execute=execute)
        self.routes = self._routes()

    # -- lifecycle ---------------------------------------------------------

    def start_workers(self) -> None:
        self.pool.start()

    def stop_workers(self) -> None:
        self.pool.stop()

    # -- routing -----------------------------------------------------------

    def _routes(self) -> list[Route]:
        experiment = r"(?P<experiment_id>[A-Za-z0-9_.\-]+)"
        return [
            Route("GET", re.compile(r"^/health$"), self.health,
                  "liveness, version, queue depth and worker state"),
            Route("GET", re.compile(r"^/api$"), self.describe_api,
                  "this list"),
            Route("GET", re.compile(r"^/experiments$"), self.list_experiments,
                  "every experiment in the workspace"),
            Route("POST", re.compile(r"^/experiments$"), self.create_experiment,
                  "submit a YAML specification; validates and plans, never runs"),
            Route("GET", re.compile(rf"^/experiments/{experiment}$"),
                  self.get_experiment, "one experiment's setup and status"),
            Route("GET", re.compile(rf"^/experiments/{experiment}/spec$"),
                  self.get_spec, "the exact YAML it was created from"),
            Route("POST", re.compile(rf"^/experiments/{experiment}/run$"),
                  self.run_experiment, "queue a run; requires --execute"),
            Route("POST", re.compile(rf"^/experiments/{experiment}/reproduce$"),
                  self.reproduce_experiment,
                  "queue a reproduction; requires --execute"),
            Route("GET", re.compile(rf"^/experiments/{experiment}/runs$"),
                  self.list_runs, "every run, with metrics and resources"),
            Route("GET", re.compile(rf"^/experiments/{experiment}/runs/(?P<run_id>[^/]+)$"),
                  self.get_run, "one run in full"),
            Route("GET", re.compile(rf"^/experiments/{experiment}/analysis$"),
                  self.get_analysis, "the complete statistical analysis"),
            Route("GET", re.compile(rf"^/experiments/{experiment}/graph$"),
                  self.get_graph, "the provenance graph, or one node's lineage"),
            Route("GET", re.compile(rf"^/experiments/{experiment}/report$"),
                  self.get_report, "a rendered report: ?format=html|md|tex|json"),
            Route("GET", re.compile(rf"^/experiments/{experiment}/certificate$"),
                  self.get_certificate, "a reproduction certificate"),
            Route("GET", re.compile(rf"^/experiments/{experiment}/why$"),
                  self.get_why, "trace a number: ?metric=&seed=&condition="),
            Route("GET", re.compile(rf"^/experiments/{experiment}/impact$"),
                  self.get_impact, "what depends on a node: ?node="),
            Route("GET", re.compile(r"^/diff$"), self.get_diff,
                  "differences between two experiments: ?left=&right="),
            Route("GET", re.compile(r"^/benchmarks$"), self.list_benchmarks,
                  "benchmark definitions on the search path"),
            Route("GET", re.compile(r"^/jobs$"), self.list_jobs,
                  "the queue: ?experiment=&state=&limit="),
            Route("GET", re.compile(r"^/jobs/(?P<job_id>[^/]+)$"), self.get_job,
                  "one job's state and result"),
            Route("POST", re.compile(r"^/jobs/(?P<job_id>[^/]+)/cancel$"),
                  self.cancel_job, "cancel a job that has not started"),
        ]

    def dispatch(self, method: str, path: str, query: dict[str, list[str]],
                 body: bytes) -> tuple[int, str, str | bytes]:
        """Route one request. Returns ``(status, media type, body)``."""
        for route in self.routes:
            if route.method != method:
                continue
            match = route.pattern.match(path)
            if match is None:
                continue
            try:
                return route.handler(query=query, body=body, **match.groupdict())
            except service.ServiceError as exc:
                return self._json(exc.status, exc.to_dict())
            except Exception as exc:
                # The trace goes to the process log, not to the client: an
                # API that returns tracebacks tells an attacker the layout
                # of the filesystem it runs on.
                traceback.print_exc()
                return self._json(500, {
                    "error": f"{type(exc).__name__}: {exc}",
                    "status": 500,
                    "remedy": "See the server log for the traceback.",
                })

        allowed = sorted({r.method for r in self.routes if r.pattern.match(path)})
        if allowed:
            return self._json(405, {
                "error": f"{method} is not allowed on {path}",
                "status": 405,
                "remedy": f"Try {', '.join(allowed)}.",
            })
        return self._json(404, {
            "error": f"no route for {method} {path}",
            "status": 404,
            "remedy": "GET /api lists every route.",
        })

    # -- helpers -----------------------------------------------------------

    @staticmethod
    def _json(status: int, payload: Any) -> tuple[int, str, str]:
        return (status, "application/json; charset=utf-8",
                json_dumps(payload))

    @staticmethod
    def _one(query: dict[str, list[str]], key: str, default: str = "") -> str:
        values = query.get(key) or []
        return values[0] if values else default

    def _resolve(self, reference: str) -> str:
        return service.resolve_experiment(self.workspace, reference)

    def _decode(self, body: bytes) -> Any:
        if not body:
            return {}
        try:
            return json.loads(body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise service.ServiceError(
                f"the request body is not valid JSON: {exc}", 400) from exc

    # -- handlers ----------------------------------------------------------

    def health(self, **_: Any) -> tuple[int, str, str]:
        counts = self.jobs.counts()
        return self._json(200, {
            "status": "ok",
            "fabric_version": __version__,
            "api_version": API_VERSION,
            "uptime_seconds": time.time() - self.started_at,
            "workspace": self.workspace.root,
            "execution_enabled": self.execute,
            "jobs": counts,
            "workers": self.pool.status(),
        })

    def describe_api(self, **_: Any) -> tuple[int, str, str]:
        return self._json(200, {
            "api_version": API_VERSION,
            "fabric_version": __version__,
            "execution_enabled": self.execute,
            "routes": [
                {"method": r.method, "path": r.pattern.pattern,
                 "description": r.description}
                for r in self.routes
            ],
        })

    def list_experiments(self, *, query: dict[str, list[str]],
                         **_: Any) -> tuple[int, str, str]:
        limit = int(self._one(query, "limit", "50") or 50)
        return self._json(200, {
            "experiments": service.list_experiments(self.workspace, limit=limit),
        })

    def create_experiment(self, *, body: bytes, **_: Any) -> tuple[int, str, str]:
        """Accept a specification. Plans it; never executes it.

        The body may be the YAML itself, or JSON with a ``spec`` field. The
        YAML form is what ``curl --data-binary @experiment.yaml`` sends,
        which is how anyone will actually try it first.
        """
        text = body.decode("utf-8", errors="replace")
        stripped = text.lstrip()
        if stripped.startswith("{"):
            payload = self._decode(body)
            text = str(payload.get("spec") or payload.get("yaml") or "")
        if not text.strip():
            raise service.ServiceError(
                "no specification in the request body", 400,
                remedy='Send the YAML directly, or JSON of the form {"spec": "..."}.',
            )
        result = service.submit_specification(self.workspace, text)
        return self._json(201, result)

    def get_experiment(self, *, experiment_id: str, **_: Any) -> tuple[int, str, str]:
        resolved = self._resolve(experiment_id)
        return self._json(200, service.describe_experiment(self.workspace, resolved))

    def get_spec(self, *, experiment_id: str, **_: Any) -> tuple[int, str, str]:
        resolved = self._resolve(experiment_id)
        text = service.specification_text(self.workspace, resolved)
        return (200, "text/yaml; charset=utf-8", text)

    def run_experiment(self, *, experiment_id: str, body: bytes,
                       **_: Any) -> tuple[int, str, str]:
        resolved = self._resolve(experiment_id)
        if not self.execute:
            raise service.ServiceError(
                "this server will not execute submitted code", 403,
                remedy="It was started without --execute. Running an experiment "
                       "means importing and calling the code its specification "
                       "names, so Fabric requires that to be switched on "
                       "deliberately. Start it with 'fabric serve --execute' if "
                       "the submissions are trusted, and read the threat model in "
                       "docs/deployment.md first.",
            )
        payload = self._decode(body)
        plan = service.load_plan(self.workspace, resolved)
        job = self.jobs.enqueue("run", resolved, payload=payload,
                                total=len(plan.runs))
        return self._json(202, {"job": job.to_dict(),
                                "poll": f"/jobs/{job.id}"})

    def reproduce_experiment(self, *, experiment_id: str, body: bytes,
                             **_: Any) -> tuple[int, str, str]:
        resolved = self._resolve(experiment_id)
        if not self.execute:
            raise service.ServiceError(
                "this server will not execute submitted code", 403,
                remedy="Reproduction re-runs the experiment, so it needs the same "
                       "permission as running it. Start with 'fabric serve --execute'.",
            )
        payload = self._decode(body)
        job = self.jobs.enqueue("reproduce", resolved, payload=payload)
        return self._json(202, {"job": job.to_dict(), "poll": f"/jobs/{job.id}"})

    def list_runs(self, *, experiment_id: str, query: dict[str, list[str]],
                  **_: Any) -> tuple[int, str, str]:
        resolved = self._resolve(experiment_id)
        return self._json(200, {
            "experiment_id": resolved,
            "runs": service.list_runs(self.workspace, resolved,
                                      condition=self._one(query, "condition")),
        })

    def get_run(self, *, experiment_id: str, run_id: str,
                **_: Any) -> tuple[int, str, str]:
        resolved = self._resolve(experiment_id)
        return self._json(200, service.describe_run(self.workspace, resolved, run_id))

    def get_analysis(self, *, experiment_id: str, **_: Any) -> tuple[int, str, str]:
        resolved = self._resolve(experiment_id)
        return self._json(200, service.analysis_payload(self.workspace, resolved))

    def get_graph(self, *, experiment_id: str, query: dict[str, list[str]],
                  **_: Any) -> tuple[int, str, str]:
        resolved = self._resolve(experiment_id)
        return self._json(200, service.graph_payload(
            self.workspace, resolved, node=self._one(query, "node")))

    def get_report(self, *, experiment_id: str, query: dict[str, list[str]],
                   **_: Any) -> tuple[int, str, str | bytes]:
        resolved = self._resolve(experiment_id)
        content, media = service.report_payload(
            self.workspace, resolved, fmt=self._one(query, "format", "json"))
        return (200, media, content)

    def get_certificate(self, *, experiment_id: str, **_: Any) -> tuple[int, str, str]:
        resolved = self._resolve(experiment_id)
        return self._json(200, service.certificate_payload(self.workspace, resolved))

    def get_why(self, *, experiment_id: str, query: dict[str, list[str]],
                **_: Any) -> tuple[int, str, str]:
        resolved = self._resolve(experiment_id)
        seed_text = self._one(query, "seed")
        try:
            seed = int(seed_text) if seed_text else None
        except ValueError as exc:
            raise service.ServiceError(f"seed must be an integer, got {seed_text!r}",
                                       400) from exc
        return self._json(200, service.why_payload(
            self.workspace, resolved, metric=self._one(query, "metric"),
            seed=seed, condition=self._one(query, "condition")))

    def get_impact(self, *, experiment_id: str, query: dict[str, list[str]],
                   **_: Any) -> tuple[int, str, str]:
        resolved = self._resolve(experiment_id)
        return self._json(200, service.impact_payload(
            self.workspace, resolved, node=self._one(query, "node")))

    def get_diff(self, *, query: dict[str, list[str]], **_: Any) -> tuple[int, str, str]:
        left, right = self._one(query, "left"), self._one(query, "right")
        if not left or not right:
            raise service.ServiceError(
                "both 'left' and 'right' are required", 400,
                remedy="GET /diff?left=EXP-a&right=EXP-b",
            )
        return self._json(200, service.diff_payload(self.workspace, left, right))

    def list_benchmarks(self, **_: Any) -> tuple[int, str, str]:
        return self._json(200, {
            "benchmarks": service.benchmarks_payload(self.workspace.root),
        })

    def list_jobs(self, *, query: dict[str, list[str]], **_: Any) -> tuple[int, str, str]:
        jobs = self.jobs.list(
            experiment_id=self._one(query, "experiment"),
            state=self._one(query, "state"),
            limit=int(self._one(query, "limit", "100") or 100),
        )
        return self._json(200, {"jobs": [j.to_dict() for j in jobs],
                                "counts": self.jobs.counts()})

    def get_job(self, *, job_id: str, **_: Any) -> tuple[int, str, str]:
        job = self.jobs.get(job_id)
        if job is None:
            raise service.ServiceError(f"no job {job_id}", 404)
        return self._json(200, job.to_dict())

    def cancel_job(self, *, job_id: str, **_: Any) -> tuple[int, str, str]:
        job = self.jobs.get(job_id)
        if job is None:
            raise service.ServiceError(f"no job {job_id}", 404)
        if not self.jobs.cancel(job_id):
            raise service.ServiceError(
                f"job {job_id} is {job.state} and cannot be cancelled", 409,
                remedy="Only a queued job can be cancelled. A running job owns a "
                       "subprocess and a partly written run directory, and killing "
                       "it from outside is how workspaces get corrupted.",
            )
        return self._json(200, (self.jobs.get(job_id) or job).to_dict())


# --------------------------------------------------------------------------
# the HTTP plumbing
# --------------------------------------------------------------------------


class _Handler(BaseHTTPRequestHandler):
    server_version = f"ExperimentFabric/{__version__}"
    sys_version = ""
    application: FabricServer

    # -- request handling --------------------------------------------------

    def do_GET(self) -> None:
        self._handle("GET")

    def do_POST(self) -> None:
        self._handle("POST")

    def do_HEAD(self) -> None:
        self._handle("GET", head_only=True)

    def _handle(self, method: str, *, head_only: bool = False) -> None:
        parsed = urllib.parse.urlparse(self.path)
        path = urllib.parse.unquote(parsed.path).rstrip("/") or "/"
        query = urllib.parse.parse_qs(parsed.query)

        refusal = self._cross_origin_refusal(method)
        if refusal is not None:
            self._refuse(refusal)
            return

        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY_BYTES:
            self.close_connection = True
            if length <= DRAIN_LIMIT_BYTES:
                self._drain(length)
            self._respond(413, "application/json; charset=utf-8", json.dumps({
                "error": f"request body is {length} bytes, and the limit is "
                         f"{MAX_BODY_BYTES}",
                "status": 413,
                "remedy": "An experiment specification is kilobytes. If yours is "
                          "not, the data belongs in a dataset the specification "
                          "references by digest, not in the specification itself.",
            }))
            return
        body = self.rfile.read(length) if length > 0 else b""

        if method == "GET" and self.application.serve_ui:
            served = self._maybe_static(path, head_only=head_only)
            if served:
                return

        status, media, content = self.application.dispatch(method, path, query, body)
        self._respond(status, media, content, head_only=head_only)

    def _cross_origin_refusal(self, method: str) -> dict[str, object] | None:
        """Refuse a state-changing request that a web page drove here.

        Binding to 127.0.0.1 is not a trust boundary against a browser. Any
        page the researcher visits can POST to this server, and a plain form
        needs no preflight, so without a check a drive-by page can queue
        execution of a specification it supplied. That is only reachable
        with ``--execute``, but then it is code execution on this machine
        from a page nobody suspects.

        Two signals are used, because each alone is defeatable:

        * ``Sec-Fetch-Site`` when the browser sends it (all current ones do).
          A request from a page the researcher is looking at is
          ``cross-site`` or ``same-site``; one from Fabric's own interface
          is ``same-origin``.
        * ``Origin``/``Referer`` otherwise, compared to the host header.

        A request with neither header is refused rather than allowed: a
        genuine API client can send ``Origin``, and one that cannot should
        be using the CLI, which does not go through HTTP at all.
        """
        if method != "POST":
            return None
        site = (self.headers.get("Sec-Fetch-Site") or "").strip().lower()
        if site:
            if site in ("same-origin", "none"):
                return None
            return _CSRF_REFUSAL

        host = (self.headers.get("Host") or "").strip().lower()
        origin = (self.headers.get("Origin") or "").strip()
        referer = (self.headers.get("Referer") or "").strip()
        candidate = origin or referer
        if not candidate:
            return _CSRF_REFUSAL
        try:
            parsed = urllib.parse.urlparse(candidate)
            candidate_host = parsed.netloc.lower()
        except ValueError:
            return _CSRF_REFUSAL
        if not candidate_host:
            return _CSRF_REFUSAL
        if host and candidate_host != host:
            return _CSRF_REFUSAL
        if not host and parsed.scheme not in ("http", "https"):
            return _CSRF_REFUSAL
        return None

    def _refuse(self, refusal: dict[str, object]) -> None:
        """Refuse without reading the body, so a refused POST cannot be a DoS."""
        self.close_connection = True
        self._respond(403, "application/json; charset=utf-8", json.dumps(refusal))

    def _drain(self, length: int) -> None:
        """Read and discard a body we are about to refuse."""
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(65536, remaining))
            if not chunk:
                return
            remaining -= len(chunk)

    def _maybe_static(self, path: str, *, head_only: bool = False) -> bool:
        """Serve the interface for paths the API does not claim."""
        if path == "/":
            relative = "index.html"
        elif path.startswith("/ui/"):
            relative = path[len("/ui/"):]
        else:
            return False

        # Resolve and confirm the result is still inside the UI directory:
        # without this, /ui/../../etc/passwd is a file read.
        candidate = os.path.realpath(os.path.join(UI_DIRECTORY, relative))
        root = os.path.realpath(UI_DIRECTORY)
        if not candidate.startswith(root + os.sep) and candidate != root:
            self._respond(403, "application/json; charset=utf-8",
                          json.dumps({"error": "forbidden", "status": 403}))
            return True
        if not os.path.isfile(candidate):
            return False
        with open(candidate, "rb") as handle:
            payload = handle.read()
        media = {
            ".html": "text/html; charset=utf-8",
            ".css": "text/css; charset=utf-8",
            ".js": "text/javascript; charset=utf-8",
            ".svg": "image/svg+xml",
            ".json": "application/json; charset=utf-8",
        }.get(os.path.splitext(candidate)[1], "application/octet-stream")
        self._respond(200, media, payload, head_only=head_only)
        return True

    def _respond(self, status: int, media: str, content: str | bytes, *,
                 head_only: bool = False) -> None:
        payload = content.encode("utf-8") if isinstance(content, str) else content
        self.send_response(status)
        self.send_header("Content-Type", media)
        self.send_header("Content-Length", str(len(payload)))
        # The UI is served from the same origin and nothing here is
        # designed to be embedded, so deny both rather than leaving the
        # defaults to whatever the browser assumes.
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("X-Frame-Options", "DENY")
        self.end_headers()
        if not head_only:
            try:
                self.wfile.write(payload)
            except (BrokenPipeError, ConnectionResetError):  # pragma: no cover
                pass  # the client went away mid-response; nothing to do

    # -- logging -----------------------------------------------------------

    def log_message(self, format: str, *args: Any) -> None:
        """One structured line per request (section 37)."""
        print(json.dumps({
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "client": self.address_string(),
            "message": format % args,
        }), flush=True)


class _HTTPServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def build_server(workspace: Workspace, *, host: str = "127.0.0.1", port: int = 8080,
                 execute: bool = False, workers: int = 1,
                 database: str = "", serve_ui: bool = True) -> tuple[Any, FabricServer]:
    """Construct the HTTP server and the application behind it.

    Returned separately so tests can drive :class:`FabricServer` directly
    without a socket, and so the caller decides when to start serving.
    """
    application = FabricServer(workspace, execute=execute, workers=workers,
                               database=database, serve_ui=serve_ui)
    handler = type("_BoundHandler", (_Handler,), {"application": application})
    httpd = _HTTPServer((host, port), handler)
    return httpd, application


def serve(workspace: Workspace, *, host: str = "127.0.0.1", port: int = 8080,
          execute: bool = False, workers: int = 1, database: str = "",
          serve_ui: bool = True,
          announce: Callable[[str], None] | None = None) -> None:
    """Run the server until interrupted."""
    httpd, application = build_server(workspace, host=host, port=port,
                                      execute=execute, workers=workers,
                                      database=database, serve_ui=serve_ui)
    application.start_workers()
    say = announce or (lambda line: print(line, flush=True))
    bound = httpd.server_address
    say(f"Experiment Fabric {__version__} listening on http://{bound[0]}:{bound[1]}")
    say(f"  workspace   {workspace.root}")
    say(f"  workers     {workers}")
    say(f"  execution   {'ENABLED' if execute else 'disabled'}")
    if execute:
        say("  WARNING: this server will import and run the code that submitted")
        say("  specifications name, as this user, with no sandbox. Only expose it")
        say("  to people you would give a shell to. See docs/deployment.md.")
    else:
        say("  submitted specifications are validated and planned, never executed;")
        say("  pass --execute to change that, after reading docs/deployment.md")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:  # pragma: no cover - interactive
        say("")
        say("stopping; running jobs are left to finish their current run")
    finally:
        httpd.shutdown()
        httpd.server_close()
        application.stop_workers()


def free_port() -> int:
    """An unused localhost port, for tests and for ``--port 0``."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


@dataclass
class BackgroundServer:
    """A server running in a thread, for tests and for ``docker compose``."""

    httpd: Any
    application: FabricServer
    thread: threading.Thread

    @property
    def url(self) -> str:
        host, port = self.httpd.server_address[:2]
        return f"http://{host}:{port}"

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()
        self.application.stop_workers()
        self.thread.join(timeout=5)


def start_background(workspace: Workspace, **kwargs: Any) -> BackgroundServer:
    """Start a server in a daemon thread and return a handle to it."""
    httpd, application = build_server(workspace, **kwargs)
    application.start_workers()
    thread = threading.Thread(target=httpd.serve_forever, daemon=True,
                              name="fabric-http")
    thread.start()
    return BackgroundServer(httpd=httpd, application=application, thread=thread)
