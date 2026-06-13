# whisper-service

Офлайн-распознавание речи на Rust + whisper.cpp. Данные никуда не уходят.

## Быстрый старт

### 1. Скачать модель

```bash
mkdir models

# Маленькая (~150MB), быстрая, достаточно для тестов
curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin \
     -o models/ggml-small.bin

# Средняя (~1.5GB) — лучшее качество для русского
curl -L https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-medium.bin \
     -o models/ggml-medium.bin
```

Доступные модели: `tiny`, `base`, `small`, `medium`, `large-v3`

### 2. Собрать и запустить

```bash
# Нужен cmake и clang (для компиляции whisper.cpp)
sudo apt install cmake clang libclang-dev   # Ubuntu/Debian
brew install cmake                           # macOS

cargo build --release

WHISPER_MODEL=models/ggml-small.bin \
WHISPER_LANG=ru \
BIND_ADDR=0.0.0.0:3000 \
./target/release/whisper-service
```

### 3. Через Docker

```bash
docker build -t whisper-service .

docker run -d \
  -p 3000:3000 \
  -v $(pwd)/models:/app/models \
  -e WHISPER_MODEL=models/ggml-medium.bin \
  -e WHISPER_LANG=ru \
  whisper-service
```

## API

### `GET /health`
```json
{ "status": "ok", "model": "ru" }
```

### `POST /transcribe`
Принимает `multipart/form-data` с аудио-файлом (WAV, 16kHz, mono).

```bash
curl -X POST http://localhost:3000/transcribe \
  -F "file=@audio.wav"
```

Ответ:
```json
{
  "text": "привет, это тест распознавания речи",
  "language": "ru",
  "duration_ms": 1240
}
```

## Конвертация аудио (ffmpeg)

Whisper.cpp ожидает WAV 16kHz mono. Конвертируй любой формат:

```bash
# WebM/Opus (браузер MediaRecorder) → WAV
ffmpeg -i input.webm -ar 16000 -ac 1 -c:a pcm_s16le output.wav

# MP3 → WAV  
ffmpeg -i input.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```

Или добавить ресамплинг прямо в сервис через крейт `rubato`.

## Переменные окружения

| Переменная      | По умолчанию              | Описание                        |
|----------------|--------------------------|----------------------------------|
| `WHISPER_MODEL` | `models/ggml-medium.bin` | Путь к модели                   |
| `WHISPER_LANG`  | `ru`                     | Язык (`ru`, `en`, `auto`, ...)  |
| `BIND_ADDR`     | `0.0.0.0:3000`           | Адрес и порт                    |

## Интеграция в Cargo workspace

В `Cargo.toml` воркспейса:
```toml
[workspace]
members = [
    "gateway",
    "auth",
    "whisper-service",
    # ...
]
```
