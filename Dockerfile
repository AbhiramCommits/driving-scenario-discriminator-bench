# Multi-stage: compile the extension + install all dependencies once, then
# copy the resulting environment into a slim runtime that only carries the
# benchmark scripts (the dsdbench source tree is deliberately NOT copied, so
# `import dsdbench` resolves to the installed package).

FROM python:3.11-slim AS builder
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake ninja-build && \
    rm -rf /var/lib/apt/lists/*
WORKDIR /repo
COPY pyproject.toml README.md README.template.md Makefile .coveragerc ./
COPY dsdbench dsdbench
COPY src src
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir ".[ml,eval]"

FROM python:3.11-slim AS runtime
RUN apt-get update && apt-get install -y --no-install-recommends make && \
    rm -rf /var/lib/apt/lists/*
COPY --from=builder /usr/local/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin
WORKDIR /repo
# Benchmark scripts + artifacts only; dsdbench/ is served from site-packages.
COPY configs configs
COPY experiments experiments
COPY scripts scripts
COPY artifacts artifacts
COPY Makefile ./
COPY README.md README.template.md ./
CMD ["make", "reproduce-quick"]
