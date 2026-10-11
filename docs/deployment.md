# Deployment

Fabric is local-first. A researcher never needs a server: `fabric run`,
`fabric analyze` and `fabric export` work against a directory, and the
evidence they produce is plain files that outlive the tool.

The server exists for the cases where that is not enough — a shared
workspace a group can browse, results visible to someone without a
terminal, a queue that survives a restart. It is the *same code*: the API
and the CLI both call `fabric.server.service`, and neither contains logic
the other cannot reach.

---

## 1. Run it

### Locally

```bash
fabric init                      # if there is no workspace yet
fabric serve                     # http://127.0.0.1:8080
```

That server **will not execute anything**. It parses, expands, hashes and
stores submitted specifications, and serves every analysis, report,
graph and certificate for experiments already in the workspace. The run
endpoint returns `403` with an explanation.

To let it run experiments:

```bash
fabric serve --execute --i-understand
```

Read [section 3](#3-the-threat-model) before you do. The second flag is
there because the first one deserves a moment's thought.

### In a container

```bash
docker compose up --build
open http://localhost:8080
```

The workspace is a named volume, so evidence survives `docker rm`. The
port is bound to `127.0.0.1` — Fabric has no authentication and no TLS,
so anything public belongs behind a reverse proxy that has both.

With execution on, and the network cut off:

```bash
docker compose --profile execute up --build   # http://localhost:8081
```

---

## 2. What the API offers

Every route returns JSON unless it says otherwise. `GET /api` lists them
at runtime, which is the version that cannot go stale.

| Method | Path | What it does |
|---|---|---|
| `GET` | `/health` | version, queue depth, worker state, whether execution is on |
| `GET` | `/api` | every route, with a description |
| `GET` | `/experiments` | every experiment, with status and run counts |
| `POST` | `/experiments` | submit a specification: validates and plans, never runs |
| `GET` | `/experiments/{id}` | setup, conditions, inputs, benchmark conformance |
| `GET` | `/experiments/{id}/spec` | the exact YAML it was created from |
| `POST` | `/experiments/{id}/run` | queue a run — **requires `--execute`** |
| `POST` | `/experiments/{id}/reproduce` | queue a reproduction — **requires `--execute`** |
| `GET` | `/experiments/{id}/runs` | every run, with metrics and resources |
| `GET` | `/experiments/{id}/runs/{run}` | one run in full |
| `GET` | `/experiments/{id}/analysis` | the complete statistical analysis |
| `GET` | `/experiments/{id}/graph` | the provenance graph, or one node's lineage |
| `GET` | `/experiments/{id}/report` | `?format=html\|md\|tex\|json` |
| `GET` | `/experiments/{id}/certificate` | a reproduction certificate |
| `GET` | `/experiments/{id}/why` | trace a number: `?metric=&seed=&condition=` |
| `GET` | `/experiments/{id}/impact` | what depends on a node: `?node=` |
| `GET` | `/diff` | `?left=&right=` |
| `GET` | `/benchmarks` | benchmark definitions on the search path |
| `GET` | `/jobs`, `/jobs/{id}` | the queue |
| `POST` | `/jobs/{id}/cancel` | cancel a job that has not started |

Submitting an experiment:

```bash
curl --data-binary @experiment.yaml http://localhost:8080/experiments
# {"experiment_id": "EXP-...", "conditions": 6, "runs": 60, "status": "planned", ...}

curl -X POST http://localhost:8080/experiments/EXP-.../run   # 403 unless --execute
```

Errors carry a `remedy` field saying what to do next, and never a
traceback — a stack trace in an HTTP response describes the filesystem of
the machine that produced it.

### Output is strictly valid JSON

Python's `json` module writes bare `NaN` and `Infinity`; nothing else
accepts them, and one of them makes an entire document unparseable to
`JSON.parse`, `jq`, Go and Rust. Fabric's analyses genuinely contain
non-finite numbers — a p-value that could not be computed, an unbounded
effect size where a condition had zero variance — so every value like
that is written as `null`. The distinction `null` loses is carried by the
`verdict`, `reason` or narrative that always travels beside it.

---

## 3. The threat model

**Running an experiment means importing and calling arbitrary Python.**
That is what a specification's `entrypoint` is. On a laptop this is
unremarkable; the code and the user are the same person. On a server that
accepts submissions it means "POST a YAML file" and "run code on this
machine" are the same operation.

### What Fabric does

| | |
|---|---|
| Execution is off by default | `FabricServer(execute=False)`; the run endpoint returns 403 |
| Turning it on is explicit | `--execute` *and* `--i-understand`, and the server says what it means at startup |
| Runs are crash-isolated | each run is a subprocess; a segfault or OOM kills one run, is classified, and the rest of the experiment survives. A submitted `isolation: inline` is overridden — the server, not the specification, decides where submitted code runs |
| Runs are time-bounded | `execution.timeout_seconds` kills a hung run. It defaults to `0`, meaning no timeout, so a specification that does not set it can hang a worker indefinitely |
| A browser cannot drive the API | every `POST` is refused unless `Sec-Fetch-Site` says same-origin, or `Origin`/`Referer` matches the host. Binding to `127.0.0.1` is not a boundary against a page the researcher visits |
| Concurrent runs are capped by the server | at most `MAX_SERVER_WORKERS` (8) per job, whatever the specification asks for. The command line has no such cap |
| Planning does not import user code | the entrypoint check is skipped on submission. With subprocess isolation it runs in the worker; see the isolation note above for the case where it is not |
| Request bodies are bounded | 1 MiB, drained and refused with a 413 rather than a connection reset |
| Static files cannot escape | `/ui/...` is resolved and confirmed to be inside the UI directory |
| Responses carry `nosniff` and `DENY` | the interface is same-origin and not meant to be framed |
| The queue survives a crash | jobs left `running` by a dead process are requeued; completed runs are skipped, so it resumes rather than repeats. Requeueing is scoped per process: a second worker on the same database will not steal a job the first one is still running |

### What Fabric does *not* do

**Subprocess isolation is not a sandbox.** The subprocess is the same
user, the same filesystem, the same network. Nothing in Fabric stops
submitted code from reading the workspace, writing outside it, opening a
socket, or deleting files it can reach.

**There is no authentication.** No users, no tokens, no permissions.
Anyone who can reach the port can do everything the API offers.

**There is no TLS.** Put a reverse proxy in front of anything that is not
on localhost.

**`ThreadingHTTPServer` is not a production web server.** No HTTP/2, no
request queuing under load, no slow-loris protection.

**A container is a boundary, not a sandbox.** `docker-compose.yml` runs
the server unprivileged, read-only, with no new privileges, all
capabilities dropped, a pids limit and a memory limit — and the `execute`
profile additionally puts it on a network with no route out. That is a
meaningfully smaller blast radius than running it on a login node. It is
not a security boundary against a determined attacker with code
execution, and a container escape is a thing that happens.

### So: when is it safe to turn execution on?

When the people who can reach the port are people you would give a shell
account to. That is the whole rule. A lab's internal machine, a
single-user deployment, a CI runner that already builds the same code —
fine. A public endpoint — no, and no combination of flags here makes it
otherwise.

If you need to accept submissions from people you do not trust, the
isolation has to come from underneath Fabric: a disposable VM or a
single-use container per run, with no credentials, no network and no
shared volume. Fabric will happily be the thing running inside it.

---

## 4. Where things are stored

Two systems, because they hold different kinds of thing.

```
metadata  →  SQLite   (.fabric/jobs.sqlite3)  — the queue
evidence  →  filesystem (.fabric/experiments) — runs, artifacts, analyses
```

A job is a small mutable row that several threads read and write and that
has to survive a restart: that is a database's job. Runs, artifacts,
analyses and bundles are immutable content-addressed files, and putting
them in a database would make them harder to verify, harder to copy and
impossible to read without the server running.

The consequence is worth stating plainly: **the database is disposable.**
Delete `jobs.sqlite3` and every experiment, run, artifact and certificate
is still there and still verifiable. What is lost is the queue. That is
the right way round for a system whose claim is that the evidence
outlives the tool.

Every query goes through `fabric/server/jobs.py` and none of them use
anything outside the SQL both SQLite and PostgreSQL understand, so moving
the metadata to Postgres is a change to one module.

Artifacts are hashed as they stream, never read whole into memory, and
recorded with size, digest and producing run. `fabric verify` re-checks
every one of them.

---

## 5. Observability

Structured logs on stdout, one JSON object per request:

```json
{"ts": "2026-09-11T21:25:44", "client": "127.0.0.1", "message": "\"GET /health HTTP/1.1\" 200 -"}
```

`GET /health` reports version, uptime, queue depth by state, per-worker
liveness, and whether execution is enabled. It is what the container
health check calls.

`GET /jobs` is the queue, filterable by experiment and state, with
progress as `completed/total` and the failure message on anything that
went wrong — a failed job carries its traceback in the job record rather
than only in a log the caller cannot see.

Deliberately not built: metrics exporters, tracing, log shipping. A first
release is better with a small real implementation than a large aspirational
one.

---

## 6. Scaling, and when not to

One process with a handful of worker threads runs experiments on one
machine. That is the intended shape, and for the workloads Fabric is for —
tens to thousands of short runs — it is enough.

The queue is durable and claiming is atomic (a conditional `UPDATE` inside
an `IMMEDIATE` transaction), so several worker processes can share one
SQLite file on one host. Across hosts, SQLite over a network filesystem is
a bad idea; that is the point at which the metadata store should become
PostgreSQL.

Deliberately not built: Kubernetes operators, autoscaling, a scheduler
with placement constraints, a GPU marketplace. Fabric's contribution is
the evidence model, and burying it under orchestration that already
exists elsewhere would not improve a single scientific claim.

---

## 7. Checking a deployment

```bash
curl -s localhost:8080/health | jq '.status, .execution_enabled'
curl -s localhost:8080/api    | jq '.routes | length'
curl --data-binary @examples/text_classification/experiment.yaml \
     localhost:8080/experiments | jq '.experiment_id, .runs'
```

The test suite does this against a real socket rather than by calling
handler functions, because routing, status codes, content types and the
security gate are exactly what a server gets wrong:

```bash
python -m pytest tests/test_server.py -v
```
