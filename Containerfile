# syntax=docker/dockerfile:1.4

FROM ghcr.io/osgeo/gdal:ubuntu-full-3.10.2 AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc-dev pkg-config

RUN --mount=type=cache,target=/root/.cargo \
    --mount=type=cache,target=/root/.rustup \
    curl --tlsv1.2 -fsS "https://sh.rustup.rs" | sh -s -- --profile minimal -y

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src src/

RUN --mount=type=cache,target=/root/.cargo \
    --mount=type=cache,target=/root/.rustup \
    --mount=type=cache,target=/app/target/release/build \
    --mount=type=cache,target=/app/target/release/deps \
    --mount=type=cache,target=/app/target/release/.fingerprint \
    ~/.cargo/bin/cargo build --release

FROM ghcr.io/osgeo/gdal:ubuntu-full-3.10.2

ARG IMAGE_SOURCE
ARG IMAGE_VERSION
ARG PACKAGE_NAME

LABEL org.opencontainers.image.source="${IMAGE_SOURCE}"
LABEL org.opencontainers.image.version="${IMAGE_VERSION}"

COPY --from=builder /app/target/release/${PACKAGE_NAME} /usr/local/bin/
