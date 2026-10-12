"""A stopped server must not keep its queue file open.

Reproduced on Windows: after `httpd.shutdown()` and `stop_workers()`,
deleting the workspace raised

    PermissionError: [WinError 32] The process cannot access the file
    because it is being used by another process: '.fabric\\jobs.sqlite3'

SQLite holds the file open for the lifetime of each connection, and
JobStore keeps one connection per thread, so stopping the workers without
closing them leaves a handle behind. On Windows that handle makes the
workspace undeletable, which breaks archiving a workspace after its server
has stopped.
"""

from __future__ import annotations

import shutil
import tempfile
import threading

from fabric.core.store import Workspace
from fabric.server.app import build_server


def test_a_stopped_server_releases_its_queue_file() -> None:
    root = tempfile.mkdtemp()
    try:
        workspace = Workspace.initialise(root)
        httpd, application = build_server(workspace, host="127.0.0.1", port=0)
        application.start_workers()
        thread = threading.Thread(target=httpd.serve_forever, daemon=True)
        thread.start()

        httpd.shutdown()
        httpd.server_close()
        application.stop_workers()
        thread.join(timeout=5)

        # The point of the test: on Windows this raised WinError 32 while the
        # worker connections were still open.
        shutil.rmtree(root)
        assert not shutil.os.path.isdir(root)
    finally:
        shutil.rmtree(root, ignore_errors=True)