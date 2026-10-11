"""The deployable layer: an HTTP API, a job queue, workers, and an interface.

Everything here is optional.  The scientific core -- specifications,
execution, statistics, analysis, provenance, reporting -- does not import
this package, and a researcher running experiments on a laptop never
touches it.  That separation is the point of section 61: the engine that
decides whether a result is real should not depend on a web server.

The dependency direction is one-way and worth stating so it stays that
way::

    fabric.core  fabric.statistics  fabric.analysis  fabric.reporting
                              ^
                              |
                       fabric.server.service
                         ^              ^
                         |              |
                   fabric.server.app   fabric.cli

:mod:`fabric.server.service` is the only place an operation is
implemented; the API and the command line both call it and neither
reimplements it.

Still no third-party dependencies: ``http.server`` and ``sqlite3`` are in
the standard library.  The interface in ``ui/`` is plain HTML, CSS and
JavaScript with no build step.
"""

from __future__ import annotations

from fabric.server.app import (
    API_VERSION,
    BackgroundServer,
    FabricServer,
    build_server,
    serve,
    start_background,
)
from fabric.server.jobs import Job, JobStore
from fabric.server.service import ServiceError
from fabric.server.worker import Worker, WorkerPool

__all__ = [
    "API_VERSION",
    "BackgroundServer",
    "FabricServer",
    "Job",
    "JobStore",
    "ServiceError",
    "Worker",
    "WorkerPool",
    "build_server",
    "serve",
    "start_background",
]
