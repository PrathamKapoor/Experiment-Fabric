# A reproduction container.
#
# The deployment story for a reproducibility tool is not "run the service";
# it is "here is a pinned environment in which this bundle re-executes".
# That is what this image is for.
#
#   docker build -t experiment-fabric .
#   docker run --rm experiment-fabric fabric doctor
#   docker run --rm -v "$PWD/bundle:/bundle" experiment-fabric fabric verify /bundle
#   docker run --rm -v "$PWD/bundle:/bundle" experiment-fabric fabric reproduce /bundle
#
# The digest of this image is worth recording in a specification's
# `algorithm.commit` or a run's environment notes: it is the only way to
# pin the environment rather than merely describe it.

FROM python:3.12-slim-bookworm

# Git is needed for commit provenance; without it Fabric records that the
# code could not be pinned, which is honest but less useful.
RUN apt-get update \
    && apt-get install --no-install-recommends -y git \
    && rm -rf /var/lib/apt/lists/*

# A non-root user, because a container that writes results as root leaves
# a workspace the host cannot read.
RUN useradd --create-home --shell /bin/bash researcher
WORKDIR /work

COPY --chown=researcher:researcher pyproject.toml README.md LICENSE ./
COPY --chown=researcher:researcher fabric ./fabric

# No dependencies to install. That is the point, and this line failing
# would mean the core had acquired one.
RUN pip install --no-cache-dir . \
    && python -c "import fabric; print('fabric', fabric.__version__)"

COPY --chown=researcher:researcher examples ./examples
COPY --chown=researcher:researcher scripts ./scripts
COPY --chown=researcher:researcher tests ./tests

USER researcher
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    FABRIC_NO_COLOR=1

# Fail the build rather than ship an image whose core quietly grew a
# dependency, or whose CLI does not start.
RUN python scripts/check_no_dependencies.py \
    && fabric --version

LABEL org.opencontainers.image.title="Experiment Fabric" \
      org.opencontainers.image.description="Reproducible, provenance-aware scientific experimentation" \
      org.opencontainers.image.licenses="Apache-2.0" \
      org.opencontainers.image.source="https://github.com/pratham/experiment-fabric"

ENTRYPOINT []
CMD ["fabric", "--help"]
