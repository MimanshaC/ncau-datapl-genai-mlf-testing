# Multi-stage Docker build with Poetry and venv
FROM python:3.10-slim as base

WORKDIR /app
# Build package from source with poetry
FROM base as builder

ENV POETRY_VERSION=1.7.0

RUN pip install "poetry==$POETRY_VERSION"

COPY pyproject.toml poetry.lock ./
RUN poetry config virtualenvs.in-project true && \
    poetry install --only=main --no-root

COPY README.md src ./
RUN poetry build

# Create final image with venv and built package installed
FROM base as final

COPY --from=builder /app/.venv ./.venv
COPY --from=builder /app/dist .
COPY src/xgb_churn_prediction/inference .

ENV PATH="./.venv/bin:$PATH"
RUN pip install *.whl
