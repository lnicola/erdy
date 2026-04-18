# syntax=docker/dockerfile:1.4

FROM ghcr.io/osgeo/gdal:ubuntu-full-3.12.1 AS builder

RUN apt-get update && \
    apt-get install -y --no-install-recommends gcc libc-dev pkg-config && \
    curl --tlsv1.2 -fsLS https://github.com/catboost/catboost/releases/download/v1.2.8/libcatboostmodel-linux-x86_64-1.2.8.so -o /usr/local/lib/libcatboostmodel-1.2.8.so && \
    chmod +x /usr/local/lib/libcatboostmodel-1.2.8.so && \
    ln -s libcatboostmodel-1.2.8.so /usr/local/lib/libcatboostmodel.so

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
    RUSTFLAGS="-L /usr/local/lib" ~/.cargo/bin/cargo build --release

FROM ghcr.io/osgeo/gdal:ubuntu-full-3.12.1

LABEL org.opencontainers.image.source="https://github.com/lnicola/erdy"
LABEL org.opencontainers.image.version="0.3.0"

RUN curl --tlsv1.2 -fsLS https://github.com/catboost/catboost/releases/download/v1.2.8/libcatboostmodel-linux-x86_64-1.2.8.so -o /usr/local/lib/libcatboostmodel-1.2.8.so && \
    chmod +x /usr/local/lib/libcatboostmodel-1.2.8.so && \
    ln -s libcatboostmodel-1.2.8.so /usr/local/lib/libcatboostmodel.so && \
    ldconfig

COPY --from=builder /app/target/release/erdy /usr/local/bin/
