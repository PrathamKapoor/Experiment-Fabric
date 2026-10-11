"""The HTTP API, the job queue and the workers, over a real socket.

These tests start an actual server and talk to it with ``urllib``. Testing
the handler functions directly would be faster and would prove less: the
things that break in a server are routing, status codes, content types and
the security gate, and none of those are exercised by calling a Python
function.
"""

from __future__ import annotations

import json
import os
import threading
import time
import urllib.error
import urllib.request

import pytest

from fabric.core.store import Workspace
from fabric.server.app import free_port, start_background
from fabric.server.jobs import QUEUED, RUNNING, SUCCEEDED, JobStore

SPEC = """
experiment:
  name: served
research_question: Does the API reach the same objects the CLI does?
algorithm:
  name: method
  entrypoint: python:tests.helpers:ablatable
components: [alpha, beta]
metrics:
  - name: accuracy
    direction: maximize
    primary: true
    tolerance: 0.02
baselines:
  - name: control
    algorithm:
      name: control
      entrypoint: python:tests.helpers:deterministic
ablations: [remove_alpha]
seeds:
  values: [1, 2, 3, 4, 5, 6]
execution:
  isolation: inline
"""


SAME_ORIGIN = {"Sec-Fetch-Site": "same-origin"}


def request(url, *, method="GET", data=None, content_type="application/json",
            headers=SAME_ORIGIN):
    """One HTTP call, returning ``(status, headers, parsed body)``.

    Requests carry ``Sec-Fetch-Site: same-origin`` by default, because that
    is what Fabric's own interface sends. The cross-origin tests replace it.
    """
    payload = data.encode("utf-8") if isinstance(data, str) else data
    req = urllib.request.Request(url, data=payload, method=method)
    if payload is not None:
        req.add_header("Content-Type", content_type)
    for key, value in (headers or {}).items():
        if value is None:
            continue
        req.add_header(key, value)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            body = response.read().decode("utf-8")
            return response.status, dict(response.headers), _parse(response, body)
    except urllib.error.HTTPError as error:
        body = error.read().decode("utf-8")
        return error.code, dict(error.headers), _parse(error, body)


def _parse(response, body):
    if "json" in (response.headers.get("Content-Type") or ""):
        return json.loads(body)
    return body


def wait_for(predicate, *, timeout=60.0, interval=0.1):
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(interval)
    raise AssertionError("condition was not met within the timeout")


# --------------------------------------------------------------------------
# fixtures
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def read_only(tmp_path_factory):
    """A server that will not execute anything: the default posture."""
    root = tmp_path_factory.mktemp("server-ro")
    cwd = os.getcwd()
    os.chdir(root)
    workspace = Workspace.initialise(str(root))
    server = start_background(workspace, host="127.0.0.1", port=free_port(),
                              execute=False, workers=1)
    try:
        yield server
    finally:
        server.stop()
        os.chdir(cwd)


@pytest.fixture(scope="module")
def executing(tmp_path_factory):
    """A server with execution switched on, as ``--execute`` would."""
    root = tmp_path_factory.mktemp("server-rw")
    cwd = os.getcwd()
    os.chdir(root)
    workspace = Workspace.initialise(str(root))
    server = start_background(workspace, host="127.0.0.1", port=free_port(),
                              execute=True, workers=1)
    try:
        yield server
    finally:
        server.stop()
        os.chdir(cwd)


@pytest.fixture(scope="module")
def submitted(read_only):
    status, _headers, body = request(f"{read_only.url}/experiments",
                                     method="POST", data=SPEC,
                                     content_type="text/yaml")
    assert status == 201, body
    return body["experiment_id"]


# --------------------------------------------------------------------------
# health and discovery
# --------------------------------------------------------------------------


def test_health_reports_version_queue_and_whether_it_will_execute(read_only):
    status, headers, body = request(f"{read_only.url}/health")
    assert status == 200
    assert "application/json" in headers["Content-Type"]
    assert body["status"] == "ok"
    assert body["fabric_version"]
    assert body["api_version"]
    assert body["execution_enabled"] is False
    assert set(body["jobs"]) >= {"queued", "running", "succeeded", "failed"}
    assert body["workers"] and body["workers"][0]["alive"] is True


def test_the_api_describes_its_own_routes(read_only):
    _status, _headers, body = request(f"{read_only.url}/api")
    paths = {route["path"] for route in body["routes"]}
    assert any("experiments" in path for path in paths)
    assert all(route["description"] for route in body["routes"]), (
        "an undocumented route is an undiscoverable one"
    )


def test_an_unknown_route_says_where_to_look(read_only):
    status, _headers, body = request(f"{read_only.url}/nonsense")
    assert status == 404
    assert "/api" in body["remedy"]


def test_the_wrong_method_lists_the_right_ones(read_only, submitted):
    status, _headers, body = request(
        f"{read_only.url}/experiments/{submitted}", method="POST", data="{}")
    assert status == 405
    assert "GET" in body["remedy"]


# --------------------------------------------------------------------------
# submission
# --------------------------------------------------------------------------


def test_a_specification_can_be_submitted_as_raw_yaml(submitted):
    assert submitted.startswith("EXP-")


def test_submission_plans_but_does_not_execute(read_only, submitted):
    _status, _headers, body = request(f"{read_only.url}/experiments/{submitted}")
    assert body["status"] == "planned"
    assert body["runs"]["completed"] == 0
    assert body["runs"]["planned"] == 18  # 3 conditions x 6 seeds


def test_a_specification_can_be_submitted_as_json(read_only):
    status, _headers, body = request(
        f"{read_only.url}/experiments", method="POST",
        data=json.dumps({"spec": SPEC.replace("name: served", "name: served-json")}))
    assert status == 201
    assert body["name"] == "served-json"


def test_an_invalid_specification_is_refused_with_its_location(read_only):
    status, _headers, body = request(
        f"{read_only.url}/experiments", method="POST",
        data="experiment: {name: broken}\nmetrics: []\n", content_type="text/yaml")
    assert status == 422
    assert "error" in body
    assert body["remedy"]


def test_an_empty_body_is_refused_helpfully(read_only):
    status, _headers, body = request(f"{read_only.url}/experiments",
                                     method="POST", data="")
    assert status == 400
    assert "spec" in body["remedy"]


def test_an_oversized_body_is_refused_before_it_is_parsed(read_only):
    status, _headers, _body = request(
        f"{read_only.url}/experiments", method="POST",
        data="#" * (1 << 21), content_type="text/yaml")
    assert status == 413


# --------------------------------------------------------------------------
# the security gate
# --------------------------------------------------------------------------


def test_a_server_without_execute_refuses_to_run_submitted_code(read_only, submitted):
    status, _headers, body = request(f"{read_only.url}/experiments/{submitted}/run",
                                     method="POST", data="{}")
    assert status == 403
    assert "will not execute" in body["error"]
    assert "--execute" in body["remedy"]
    assert "docs/deployment.md" in body["remedy"]


def test_reproduction_is_gated_by_the_same_permission(read_only, submitted):
    status, _headers, body = request(
        f"{read_only.url}/experiments/{submitted}/reproduce", method="POST", data="{}")
    assert status == 403
    assert "same permission" in body["remedy"]


def test_the_gate_is_reported_in_health_so_it_is_not_a_surprise(read_only, executing):
    assert request(f"{read_only.url}/health")[2]["execution_enabled"] is False
    assert request(f"{executing.url}/health")[2]["execution_enabled"] is True


# --------------------------------------------------------------------------
# execution
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def executed(executing):
    """Submit and run one experiment on the executing server."""
    _status, _headers, created = request(f"{executing.url}/experiments",
                                         method="POST", data=SPEC,
                                         content_type="text/yaml")
    experiment_id = created["experiment_id"]
    status, _headers, queued = request(
        f"{executing.url}/experiments/{experiment_id}/run", method="POST", data="{}")
    assert status == 202, queued
    job_id = queued["job"]["id"]

    def finished():
        job = request(f"{executing.url}/jobs/{job_id}")[2]
        return job if job["state"] in ("succeeded", "failed") else None

    job = wait_for(finished, timeout=180)
    assert job["state"] == "succeeded", job
    return experiment_id, job


def test_a_queued_run_executes_and_records_what_it_did(executed):
    _experiment_id, job = executed
    assert job["result"]["completed"] == 18
    assert job["result"]["failed"] == 0
    assert job["progress"] == job["total"] == 18
    assert job["duration_seconds"] >= 0
    assert job["worker"]


def test_the_experiment_is_complete_afterwards(executing, executed):
    experiment_id, _job = executed
    body = request(f"{executing.url}/experiments/{experiment_id}")[2]
    assert body["status"] == "completed"


def test_a_submitted_spec_cannot_choose_to_run_in_the_server_process():
    """``isolation: inline`` would run submitted code here, not in a worker.

    Every other execution mode gets a crash boundary and a timeout. Inline
    execution has neither, so one submitted specification could stop the
    service serving for the rest of its life -- and the server's own
    docstring promises the opposite. The server decides this, not the
    specification, because the specification arrived over the network.
    """
    from fabric.core.spec import ExperimentSpec
    from fabric.server.worker import _options_from

    inline = ExperimentSpec.from_yaml(SPEC)
    assert inline.execution.isolation == "inline"
    assert _options_from(inline, {}).isolation == "process"

    process = ExperimentSpec.from_yaml(SPEC.replace(
        "  isolation: inline", "  isolation: process"))
    assert _options_from(process, {}).isolation == "process"


def test_a_submitted_spec_cannot_ask_for_an_unbounded_thread_pool():
    """The worker cap belongs to the server, not to the specification.

    A specification is data. Letting it choose the size of the executor
    that also serves HTTP is a denial of service with a YAML payload.
    """
    from fabric.core.spec import ExperimentSpec
    from fabric.server.worker import MAX_SERVER_WORKERS, _options_from

    greedy = ExperimentSpec.from_yaml(
        SPEC.replace("execution:", "execution:\n  max_workers: 100000"))
    assert greedy.execution.max_workers == 100000
    assert _options_from(greedy, {}).max_workers == MAX_SERVER_WORKERS

    assert _options_from(greedy, {"max_workers": 99999}).max_workers == MAX_SERVER_WORKERS
    # The specification may still ask for fewer than the cap.
    assert _options_from(greedy, {"max_workers": 2}).max_workers == 2
    assert _options_from(
        ExperimentSpec.from_yaml(SPEC), {}).max_workers <= MAX_SERVER_WORKERS


def test_an_over_eager_worker_count_is_reported_rather_than_obeyed(executing):
    """A huge ask is accepted as a job and clamped, not honoured.

    The point is that the request is not an error: the run is legitimate
    work, and the only thing refused is the unbounded executor.
    """
    status, _headers, created = request(
        f"{executing.url}/experiments", method="POST",
        data=SPEC.replace("execution:", "execution:\n  max_workers: 5000"),
        content_type="text/yaml")
    assert status == 201, created
    experiment_id = created["experiment_id"]

    status, _headers, queued = request(
        f"{executing.url}/experiments/{experiment_id}/run", method="POST",
        data=json.dumps({"max_workers": 5000}))
    assert status == 202, queued
    job_id = queued["job"]["id"]

    def finished():
        job = request(f"{executing.url}/jobs/{job_id}")[2]
        return job if job["state"] in ("succeeded", "failed") else None

    job = wait_for(finished, timeout=180)
    assert job["state"] == "succeeded", job
    assert job["result"]["completed"] == 18


def test_every_run_is_listed_with_its_seed_and_metrics(executing, executed):
    experiment_id, _job = executed
    body = request(f"{executing.url}/experiments/{experiment_id}/runs")[2]
    assert len(body["runs"]) == 18
    first = body["runs"][0]
    assert first["seed"] in range(1, 7)
    assert "accuracy" in first["metrics"]
    assert first["environment_id"] and first["hardware_id"]


def test_one_run_comes_back_in_full(executing, executed):
    experiment_id, _job = executed
    listed = request(f"{executing.url}/experiments/{experiment_id}/runs")[2]["runs"][0]
    body = request(
        f"{executing.url}/experiments/{experiment_id}/runs/{listed['id']}")[2]
    assert body["id"] == listed["id"]
    assert body["condition_name"] == listed["condition"]
    assert body["resources"]["wall_seconds"] >= 0
    assert "accuracy" in body["metrics"]


def test_the_analysis_is_the_same_object_the_cli_produces(executing, executed):
    experiment_id, _job = executed
    body = request(f"{executing.url}/experiments/{experiment_id}/analysis")[2]
    assert body["primary_metric"] == "accuracy"
    assert body["headline"]
    assert body["confidence_level"] == 0.95
    assert body["baselines"]["comparisons"]
    assert body["limitations"]


def test_the_graph_comes_back_with_typed_edges(executing, executed):
    experiment_id, _job = executed
    body = request(f"{executing.url}/experiments/{experiment_id}/graph")[2]
    assert body["nodes"] and body["edges"]
    assert all(edge.get("relation") for edge in body["edges"])


@pytest.mark.parametrize("fmt,fragment,media", [
    ("json", "title", "application/json"),
    ("md", "# ", "text/markdown"),
    ("html", "<!doctype html", "text/html"),
    ("tex", "documentclass", "application/x-tex"),
])
def test_every_report_format_renders(executing, executed, fmt, fragment, media):
    experiment_id, _job = executed
    status, headers, body = request(
        f"{executing.url}/experiments/{experiment_id}/report?format={fmt}")
    assert status == 200
    assert media in headers["Content-Type"]
    # The JSON format is parsed back into a dict by `request`; the others
    # arrive as text. Both must contain the marker, one as a key and one
    # as a substring.
    if fmt == "json":
        assert body["title"] and body["sections"]
    else:
        assert fragment.lower() in body.lower()


def test_an_unknown_report_format_is_refused(executing, executed):
    experiment_id, _job = executed
    status, _headers, body = request(
        f"{executing.url}/experiments/{experiment_id}/report?format=pdf")
    assert status == 400
    assert "html" in body["remedy"]


def test_the_certificate_is_issued_over_http(executing, executed):
    experiment_id, _job = executed
    body = request(f"{executing.url}/experiments/{experiment_id}/certificate")[2]
    assert body["checks"]
    assert body["experiment_id"] == experiment_id


def test_why_traces_a_number_back_through_the_api(executing, executed):
    experiment_id, _job = executed
    body = request(
        f"{executing.url}/experiments/{experiment_id}/why?metric=accuracy&seed=3")[2]
    assert body


def test_a_bad_seed_is_a_clear_error_not_a_stack_trace(executing, executed):
    experiment_id, _job = executed
    status, _headers, body = request(
        f"{executing.url}/experiments/{experiment_id}/why?seed=banana")
    assert status == 400
    assert "integer" in body["error"]
    assert "Traceback" not in json.dumps(body)


def test_impact_answers_what_depends_on_the_dataset(executing, executed):
    experiment_id, _job = executed
    status, _headers, body = request(
        f"{executing.url}/experiments/{experiment_id}/impact")
    # This fixture declares no dataset, so the honest answer is a 400 that
    # says so rather than a silently empty result.
    assert status == 400
    assert "no dataset" in body["error"]


def test_the_specification_comes_back_verbatim(executing, executed):
    experiment_id, _job = executed
    status, headers, body = request(f"{executing.url}/experiments/{experiment_id}/spec")
    assert status == 200
    assert "yaml" in headers["Content-Type"]
    assert body.strip() == SPEC.strip()


# --------------------------------------------------------------------------
# analysis before there is anything to analyse
# --------------------------------------------------------------------------


def test_analysing_an_unexecuted_experiment_says_what_to_do(read_only, submitted):
    status, _headers, body = request(f"{read_only.url}/experiments/{submitted}/analysis")
    assert status == 409
    assert "no runs yet" in body["error"]
    assert "run" in body["remedy"]


def test_an_unknown_experiment_is_a_404(read_only):
    status, _headers, _body = request(f"{read_only.url}/experiments/EXP-nope")
    assert status == 404


# --------------------------------------------------------------------------
# the queue
# --------------------------------------------------------------------------


def test_jobs_can_be_listed_and_filtered(executing, executed):
    experiment_id, job = executed
    body = request(f"{executing.url}/jobs?experiment={experiment_id}")[2]
    assert any(j["id"] == job["id"] for j in body["jobs"])
    assert body["counts"]["succeeded"] >= 1

    filtered = request(f"{executing.url}/jobs?state=succeeded")[2]
    assert all(j["state"] == "succeeded" for j in filtered["jobs"])


def test_an_unknown_job_is_a_404(executing):
    status, _headers, _body = request(f"{executing.url}/jobs/job-nope")
    assert status == 404


def test_a_finished_job_cannot_be_cancelled_and_says_why(executing, executed):
    _experiment_id, job = executed
    status, _headers, body = request(f"{executing.url}/jobs/{job['id']}/cancel",
                                     method="POST", data="{}")
    assert status == 409
    assert "queued job" in body["remedy"]


# --------------------------------------------------------------------------
# the interface
# --------------------------------------------------------------------------


def test_the_interface_is_served_at_the_root(read_only):
    status, headers, body = request(read_only.url + "/")
    assert status == 200
    assert "text/html" in headers["Content-Type"]
    assert "Experiment" in body


@pytest.mark.parametrize("asset,media", [("app.css", "text/css"),
                                         ("app.js", "text/javascript")])
def test_the_interface_assets_are_served(read_only, asset, media):
    status, headers, _body = request(f"{read_only.url}/ui/{asset}")
    assert status == 200
    assert media in headers["Content-Type"]


def test_the_static_handler_refuses_to_escape_its_directory(read_only):
    """``/ui/../../secrets`` must not be a file read."""
    for attempt in ("/ui/../app.py", "/ui/..%2f..%2fsetup.py",
                    "/ui/....//app.py"):
        status, _headers, _body = request(read_only.url + attempt)
        assert status in (403, 404), attempt


def test_responses_carry_the_headers_that_keep_a_browser_honest(read_only):
    _status, headers, _body = request(read_only.url + "/")
    assert headers["X-Content-Type-Options"] == "nosniff"
    assert headers["X-Frame-Options"] == "DENY"


# --------------------------------------------------------------------------
# cross-origin writes
# --------------------------------------------------------------------------


def test_a_page_cannot_queue_a_specification_of_its_own(read_only):
    """Binding to 127.0.0.1 is not a boundary against a browser.

    Any page the researcher opens can POST here, and a plain form needs no
    preflight, so without this check a drive-by page supplies a
    specification and queues it. With ``--execute`` that is code execution
    on this machine from a page nobody suspects.
    """
    status, _headers, body = request(
        f"{read_only.url}/experiments", method="POST", data=SPEC,
        content_type="text/yaml", headers={"Sec-Fetch-Site": "cross-site"})
    assert status == 403, body
    assert "cross-origin" in body["error"]


@pytest.mark.parametrize("site", ["cross-site", "same-site", "none"])
def test_sec_fetch_site_is_honoured_whenever_the_browser_sends_it(read_only, site):
    if site == "none":          # a direct navigation; not a page driving us
        assert request(f"{read_only.url}/experiments", method="POST", data=SPEC,
                       content_type="text/yaml",
                       headers={"Sec-Fetch-Site": site})[0] == 201
        return
    status, _headers, body = request(
        f"{read_only.url}/experiments", method="POST", data=SPEC,
        content_type="text/yaml", headers={"Sec-Fetch-Site": site})
    assert status == 403, body


def test_a_cross_origin_header_is_refused_when_sec_fetch_is_absent(read_only):
    """Older clients send Origin and Referer but no Sec-Fetch-Site."""
    host = read_only.url.split("//", 1)[1]
    status, _headers, body = request(
        f"{read_only.url}/experiments", method="POST", data=SPEC,
        content_type="text/yaml",
        headers={"Origin": "http://evil.example", "Referer": None})
    assert status == 403, body

    status, _headers, body = request(
        f"{read_only.url}/experiments", method="POST", data=SPEC,
        content_type="text/yaml",
        headers={"Referer": "http://evil.example/page",
                 "Sec-Fetch-Site": None})
    assert status == 403, body

    # ...and the same host is allowed, so a genuine client still works.
    status, _headers, body = request(
        f"{read_only.url}/experiments", method="POST", data=SPEC,
        content_type="text/yaml",
        headers={"Origin": f"http://{host}", "Sec-Fetch-Site": None})
    assert status == 201, body


def test_a_post_with_no_provenance_headers_at_all_is_refused(read_only):
    """Silently accepting an unattributable write is the dangerous default."""
    status, _headers, body = request(
        f"{read_only.url}/experiments", method="POST", data=SPEC,
        content_type="text/yaml", headers={})
    assert status == 403, body


def test_reads_are_never_refused(read_only, submitted):
    """Only state-changing requests are restricted; browsing must keep working."""
    assert request(f"{read_only.url}/experiments")[0] == 200
    assert request(f"{read_only.url}/experiments/{submitted}")[0] == 200
    assert request(f"{read_only.url}/health",
                   headers={"Sec-Fetch-Site": "cross-site"})[0] == 200


# --------------------------------------------------------------------------
# the job store on its own
# --------------------------------------------------------------------------


def test_claiming_is_atomic_under_concurrency(tmp_path):
    """Two workers must never believe they own the same job."""
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    for index in range(40):
        store.enqueue("run", f"EXP-{index}")

    claimed: list[str] = []
    lock = threading.Lock()

    def drain():
        local = JobStore(str(tmp_path / "jobs.sqlite3"))
        while True:
            job = local.claim(threading.current_thread().name)
            if job is None:
                return
            with lock:
                claimed.append(job.id)

    threads = [threading.Thread(target=drain, name=f"w{i}") for i in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)

    assert len(claimed) == 40
    assert len(set(claimed)) == 40, "a job was claimed twice"


def test_a_queued_job_can_be_cancelled_and_a_running_one_cannot(tmp_path):
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    queued = store.enqueue("run", "EXP-1")
    assert store.cancel(queued.id) is True
    assert (store.get(queued.id) or queued).state == "cancelled"

    running = store.enqueue("run", "EXP-2")
    store.claim("worker")
    assert store.cancel(running.id) is False
    assert (store.get(running.id) or running).state == RUNNING


def test_jobs_left_running_by_a_crash_are_requeued(tmp_path):
    """An identity with no pid cannot be judged, so it is left alone.

    Sweeping an unparseable identity would duplicate a job that a live
    sibling is running; refusing to sweep it means a crashed worker keeps
    its job, which is visible in the queue rather than silent.
    """
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    store.enqueue("run", "EXP-1")
    store.claim("worker-that-died")
    assert store.counts()[RUNNING] == 1

    assert store.requeue_running() == 0
    assert store.requeue_running(stale_after=0.0) == 1
    assert store.counts()[QUEUED] == 1
    assert store.counts()[RUNNING] == 0


def test_starting_a_second_server_does_not_steal_a_running_job(tmp_path):
    """docs/deployment.md recommends sharing one SQLite file between servers.

    An unscoped recovery sweep would make the second server requeue the
    first server's live job, and two workers would then execute the same
    runs into the same run directory at the same time. Recovery has to be
    scoped to the workers that died.
    """
    path = str(tmp_path / "jobs.sqlite3")
    store = JobStore(path)
    store.enqueue("run", "EXP-1")
    store.enqueue("run", "EXP-2")
    # This process is alive, so its job must survive another process starting.
    store.claim(f"worker-1@{os.getpid()}")
    assert store.counts()[RUNNING] == 1

    assert JobStore(path).requeue_running() == 0
    assert store.counts()[RUNNING] == 1, "a live job was stolen"
    assert JobStore(path).requeue_running() == 0


def test_a_job_whose_process_died_is_recovered(tmp_path):
    """The crash case recovery exists for, decided by process liveness."""
    import subprocess
    import sys

    dead = subprocess.Popen([sys.executable, "-c", "pass"])
    pid = dead.pid
    dead.wait()

    path = str(tmp_path / "jobs.sqlite3")
    store = JobStore(path)
    store.enqueue("run", "EXP-1")
    store.claim(f"worker-1@{pid}")
    assert store.counts()[RUNNING] == 1

    assert JobStore(path).requeue_running() == 1
    assert store.counts()[RUNNING] == 0
    assert store.counts()[QUEUED] == 1


def test_a_stale_job_can_be_recovered_by_age(tmp_path):
    """Liveness cannot see a worker that is alive but wedged."""
    path = str(tmp_path / "jobs.sqlite3")
    store = JobStore(path)
    store.enqueue("run", "EXP-1")
    store.claim(f"worker-1@{os.getpid()}")
    assert store.requeue_running(stale_after=3600) == 0
    assert store.counts()[RUNNING] == 1
    assert store.requeue_running(stale_after=0.0) == 1
    assert store.counts()[RUNNING] == 0


def test_progress_and_results_survive_a_reopen(tmp_path):
    path = str(tmp_path / "jobs.sqlite3")
    store = JobStore(path)
    job = store.enqueue("run", "EXP-1", total=10)
    store.progress(job.id, 4, 10, "halfway")
    store.finish(job.id, SUCCEEDED, message="done", result={"completed": 10})
    store.close()

    reopened = JobStore(path)
    again = reopened.get(job.id)
    assert again is not None
    assert again.progress == 4
    assert again.message == "done"
    assert again.result == {"completed": 10}
    assert again.done is True


def test_a_future_schema_version_refuses_to_reinterpret_the_data(tmp_path):
    path = str(tmp_path / "jobs.sqlite3")
    store = JobStore(path)
    store._connection.execute("UPDATE schema_version SET version = 99")
    store.close()
    with pytest.raises(Exception, match="schema version"):
        JobStore(path)
