FROM ghcr.io/osgeo/gdal:ubuntu-full-3.7.1
RUN apt-get update -y
RUN apt-get install -y binutils curl pkg-config gcc
RUN curl --proto '=https' --tlsv1.2 --retry 10 --retry-connrefused -fsSL "https://sh.rustup.rs" | sh -s -- --profile minimal --default-toolchain stable --component rust-src rustfmt clippy -y
WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src/
RUN $HOME/.cargo/bin/cargo build --release

FROM ghcr.io/osgeo/gdal:ubuntu-full-3.7.1
COPY --from=0 /app/target/release/geo-tools /usr/local/bin/
