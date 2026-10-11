# Experiment Fabric: a pinned environment, and a way to serve it.
#
# Two things, because the project is two things.
#
# 1. A REPRODUCTION CONTAINER. The deployment story for a reproducibility
#    tool is not "run the service"; it is "here is a pinned environment in
#    which this bundle re-executes". The image digest is the only way to
#    pin an environment rather than describe it, and it is worth recording
#    beside any result you intend to publish.
#
#       docker build -t experiment-fabric .
#       docker run --rm experiment-fabric fabric doctor
#       docker run --rm -v "$PWD/bundle:/bundle" experiment-fabric fabric verify /bundle
#       docker run --rm -v "$PWD/bundle:/bundle" experiment-fabric fabric reproduce /bundle
#
# 2. THE SERVICE. `docker compose up` runs the API and the interface
#    against a persistent workspace. Read docs/deployment.md before
#    turning execution on: the server runs the code that submitted
#    specifications name, and a container is a boundary, not a sandbox.
#
#       docker compose up
#       open http://localhost:8080

FROM python:3.12-slim-bookworm AS base

# Git is needed for commit provenance; without it Fabric records that the
# code could not be pinned, which is honest but less useful.
RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/*

# A non-root user, because a container that writes results as root leaves
# a workspace the host cannot read -- and because the worker executes code
# it did not write.
RUN useradd --create-home --shell /bin/bash --uid 10001 researcher
WORKDIR /work

COPY --chown=researcher:researcher pyproject.toml README.md LICENSE ./
COPY --chown=researcher:researcher fabric ./fabric

# No dependencies to install. That is the point, and this line failing
# would mean the core had acquired one.
RUN pip install --no-cache-dir . \
    && python -c "import fabric; print('fabric', fabric.__version__)"

COPY --chown=researcher:researcher examples ./examples
COPY --chown=researcher:researcher benchmarks ./benchmarks
COPY --chown=researcher:researcher scripts ./scripts
COPY --chown=researcher:researcher tests ./tests

USER researcher
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FABRIC_NO_COLOR=1 \
    FABRIC_BENCHMARK_PATH=/work/benchmarks

# Fail the build rather than ship an image whose core quietly grew a
# dependency, whose CLI does not start, or whose interface did not make it
# into the wheel.
RUN python scripts/check_no_dependencies.py \
    && fabric --version \
    && python -c "import os, fabric.server.app as a; \
assert os.path.isfile(os.path.join(a.UI_DIRECTORY, 'index.html')), \
'the web interface is missing from the installed package'; \
print('interface packaged')"

LABEL org.opencontainers.image.title="Experiment Fabric" \
      org.opencontainers.image.description="Reproducible, provenance-aware scientific experimentation" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/pratham/experiment-fabric"

ENTRYPOINT []
CMD ["fabric", "--help"]


# --------------------------------------------------------------------------
# The service.
#
# A separate stage so that `docker build --target server` produces an image
# whose default command is the API rather than the help text, and so the
# health check belongs to the thing that actually serves.
# --------------------------------------------------------------------------

FROM base AS server

# The workspace is a volume: evidence must outlive the container. Losing it
# on `docker rm` would contradict the entire premise.
VOLUME ["/data"]
ENV FABRIC_WORKSPACE=/data

EXPOSE 8080

# `fabric serve` binds localhost by default, which inside a container means
# nothing outside can reach it. 0.0.0.0 here is correct precisely because
# the container's network namespace is the boundary -- publish the port
# deliberately, and put a reverse proxy in front of anything public.
#
# Execution stays OFF. Turning it on is a decision with consequences
# (docs/deployment.md), not a default.
# --workspace is a global option and must precede the subcommand; argparse
# rejects it afterwards, which would make `docker run` fail to start the
# service for a reason that has nothing to do with the service.
CMD ["fabric", "--workspace", "/data", "serve", "--host", "0.0.0.0", \
     "--port", "8080", "--workers", "2"]

HEALTHCHECK --interval=15s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import sys, urllib.request, json; \
r = urllib.request.urlopen('http://127.0.0.1:8080/health', timeout=4); \
body = json.load(r); \
sys.exit(0 if body.get('status') == 'ok' else 1)"
