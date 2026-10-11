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


def request(url, *, method="GET", data=None, content_type="application/json"):
    """One HTTP call, returning ``(status, headers, parsed body)``."""
    payload = data.encode("utf-8") if isinstance(data, str) else data
    req = urllib.request.Request(url, data=payload, method=method)
    if payload is not None:
        req.add_header("Content-Type", content_type)
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
    assert body["runs"]["completed"] == 18


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
    store = JobStore(str(tmp_path / "jobs.sqlite3"))
    store.enqueue("run", "EXP-1")
    store.claim("worker-that-died")
    assert store.counts()[RUNNING] == 1

    assert store.requeue_running() == 1
    assert store.counts()[QUEUED] == 1
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
