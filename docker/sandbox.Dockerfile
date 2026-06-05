# Minimal, locked-down image for running untrusted, model-generated code.
# stdlib Python only -- no pip packages -- so generated code can import nothing
# we didn't sanction. The runtime hardening (network off, dropped caps, resource
# limits) is applied at `docker run` time in sandbox.py; the image just provides
# a non-root user and a stdin entrypoint (CLAUDE.md §5).
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Non-root user; this is the default runtime identity.
RUN useradd --create-home --uid 1000 sandbox
USER sandbox
WORKDIR /home/sandbox

# Read the program from stdin and execute it: `docker run -i ... < code`.
ENTRYPOINT ["python", "-"]
