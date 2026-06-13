# ── Build stage ──────────────────────────────────────────────────────────────
FROM rust:1.78-bookworm AS builder

# Нужны для компиляции whisper.cpp (C++ биндинги)
RUN apt-get update && apt-get install -y \
    cmake \
    clang \
    libclang-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY Cargo.toml Cargo.lock ./
COPY src ./src

# Собираем release билд
RUN cargo build --release

# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y \
    ca-certificates \
    libstdc++6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY --from=builder /app/target/release/whisper-service ./

# Модели кладёшь в ./models/ рядом с бинарём или монтируешь volume
RUN mkdir models

ENV WHISPER_MODEL=models/ggml-medium.bin
ENV WHISPER_LANG=ru
ENV BIND_ADDR=0.0.0.0:3000

EXPOSE 3000

CMD ["./whisper-service"]
