use anyhow::{Context, Result, anyhow};
use axum::{
    Json, Router,
    extract::{Multipart, State},
    http::StatusCode,
    response::IntoResponse,
    routing::{get, post},
};
use serde::Serialize;
use std::{
    io::Write,
    process::{Command, Stdio},
    sync::Arc,
};
use tokio::net::TcpListener;
use tokio::sync::Semaphore;
use tower_http::cors::{Any, CorsLayer};
use tower_http::trace::TraceLayer;
use tracing::info;
use whisper_rs::{FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters};

// ─── Состояние приложения ─────────────────────────────────────────────────────

struct AppState {
    whisper_ctx: WhisperContext,
    /// None = автоопределение языка whisper-ом
    language: Option<String>,
    /// Ограничивает кол-во параллельных инференсов
    semaphore: Arc<Semaphore>,
}

// WhisperContext содержит raw pointer — говорим компилятору что это безопасно.
// whisper.cpp потокобезопасен при использовании отдельного WhisperState на поток.
unsafe impl Send for AppState {}
unsafe impl Sync for AppState {}

// ─── Ответы ───────────────────────────────────────────────────────────────────

#[derive(Serialize)]
struct TranscribeResponse {
    text: String,
    language: String,
    duration_ms: u64,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

#[derive(Serialize)]
struct HealthResponse {
    status: &'static str,
    language: String,
    ffmpeg: bool,
}

// ─── Хендлеры ─────────────────────────────────────────────────────────────────

async fn health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let ffmpeg_ok = Command::new("ffmpeg").arg("-version").output().is_ok();
    Json(HealthResponse {
        status: "ok",
        language: state.language.clone().unwrap_or_else(|| "auto".into()),
        ffmpeg: ffmpeg_ok,
    })
}

async fn transcribe(
    State(state): State<Arc<AppState>>,
    mut multipart: Multipart,
) -> Result<impl IntoResponse, (StatusCode, Json<ErrorResponse>)> {
    let field = multipart
        .next_field()
        .await
        .map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?
        .ok_or_else(|| err(StatusCode::BAD_REQUEST, "no file field".into()))?;

    let bytes = field
        .bytes()
        .await
        .map_err(|e| err(StatusCode::BAD_REQUEST, e.to_string()))?
        .to_vec();

    if bytes.is_empty() {
        return Err(err(StatusCode::BAD_REQUEST, "empty audio".into()));
    }

    // Ждём свободного слота (не более MAX_CONCURRENT параллельных инференсов)
    let permit = Arc::clone(&state.semaphore)
        .acquire_owned()
        .await
        .map_err(|e| err(StatusCode::SERVICE_UNAVAILABLE, e.to_string()))?;

    let lang = state.language.clone();
    let result = tokio::task::spawn_blocking(move || {
        let _permit = permit; // освободится когда выйдем из блока
        run_whisper(&state.whisper_ctx, bytes, lang.as_deref())
    })
    .await
    .map_err(|e| err(StatusCode::INTERNAL_SERVER_ERROR, e.to_string()))?
    .map_err(|e| err(StatusCode::UNPROCESSABLE_ENTITY, e.to_string()))?;

    Ok(Json(result))
}

// ─── Whisper ──────────────────────────────────────────────────────────────────

fn run_whisper(
    ctx: &WhisperContext,
    audio_bytes: Vec<u8>,
    language: Option<&str>,
) -> Result<TranscribeResponse> {
    let start = std::time::Instant::now();

    // Декодируем через ffmpeg → PCM f32le 16kHz mono
    let samples = decode_via_ffmpeg(audio_bytes).context("failed to decode audio")?;

    info!(
        "Decoded {} samples ({:.1}s)",
        samples.len(),
        samples.len() as f32 / 16000.0
    );

    let mut state = ctx
        .create_state()
        .context("failed to create whisper state")?;

    let mut params = FullParams::new(SamplingStrategy::Greedy { best_of: 1 });
    params.set_language(language); // None = auto-detect
    params.set_print_special(false);
    params.set_print_progress(false);
    params.set_print_realtime(false);
    params.set_print_timestamps(false);
    params.set_translate(false);
    params.set_n_threads(4);

    state
        .full(params, &samples)
        .context("whisper inference failed")?;

    let n_segments = state
        .full_n_segments()
        .context("failed to get n_segments")?;
    let mut text = String::new();
    for i in 0..n_segments {
        let seg = state
            .full_get_segment_text(i)
            .context("failed to get segment text")?;
        text.push_str(&seg);
        text.push(' ');
    }
    let text = text.trim().to_string();

    // Если language=None — пробуем получить определённый язык из whisper
    let detected_lang = language.unwrap_or("auto").to_string();

    Ok(TranscribeResponse {
        text,
        language: detected_lang,
        duration_ms: start.elapsed().as_millis() as u64,
    })
}

// ─── FFmpeg декодер ───────────────────────────────────────────────────────────

/// Принимает аудио любого формата (WebM, OGG, MP3, MP4, WAV, FLAC, ...),
/// прогоняет через ffmpeg и возвращает PCM f32le 16kHz mono.
///
/// ffmpeg читает из stdin, пишет сырые сэмплы в stdout — никаких temp файлов.
fn decode_via_ffmpeg(bytes: Vec<u8>) -> Result<Vec<f32>> {
    let mut child = Command::new("ffmpeg")
        .args([
            "-hide_banner",
            "-loglevel",
            "error", // только ошибки в stderr
            "-i",
            "pipe:0", // читаем из stdin
            "-ar",
            "16000", // resample → 16kHz
            "-ac",
            "1", // downmix → mono
            "-f",
            "f32le",  // формат вывода: raw f32 little-endian
            "pipe:1", // пишем в stdout
        ])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .map_err(|e| anyhow!("failed to spawn ffmpeg: {e}. Is ffmpeg installed?"))?;

    // Пишем байты в stdin ffmpeg
    {
        let stdin = child
            .stdin
            .as_mut()
            .context("failed to open ffmpeg stdin")?;
        stdin
            .write_all(&bytes)
            .context("failed to write to ffmpeg stdin")?;
    } // stdin закрывается здесь → ffmpeg получает EOF и начинает flush

    let output = child.wait_with_output().context("ffmpeg wait failed")?;

    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr);
        return Err(anyhow!("ffmpeg error: {stderr}"));
    }

    if output.stdout.is_empty() {
        return Err(anyhow!(
            "ffmpeg produced no output — is the audio file valid?"
        ));
    }

    // Парсим raw f32le байты → Vec<f32>
    let raw = output.stdout;
    if raw.len() % 4 != 0 {
        return Err(anyhow!(
            "ffmpeg output is not aligned to f32 (len={})",
            raw.len()
        ));
    }

    let samples: Vec<f32> = raw
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
        .collect();

    info!("ffmpeg decoded {} samples", samples.len());

    Ok(samples)
}

// ─── Хелпер для ошибок ───────────────────────────────────────────────────────

fn err(code: StatusCode, msg: String) -> (StatusCode, Json<ErrorResponse>) {
    tracing::error!("{}", msg);
    (code, Json(ErrorResponse { error: msg }))
}

// ─── main ─────────────────────────────────────────────────────────────────────

#[tokio::main]
async fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("speech_recog=info".parse()?),
        )
        .init();

    let model_path =
        std::env::var("WHISPER_MODEL").unwrap_or_else(|_| "models/ggml-small.bin".to_string());

    // "auto" или не задана → None (whisper определяет сам)
    // "ru", "kk", "en" и т.д. → Some(lang)
    let language: Option<String> = match std::env::var("WHISPER_LANG").as_deref() {
        Ok("auto") | Ok("") | Err(_) => None,
        Ok(lang) => Some(lang.to_string()),
    };

    // Макс. параллельных инференсов. 1-2 для CPU, больше для GPU.
    let max_concurrent: usize = std::env::var("MAX_CONCURRENT")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(2);

    let bind_addr = std::env::var("BIND_ADDR").unwrap_or_else(|_| "0.0.0.0:3000".to_string());

    // Проверяем ffmpeg при старте
    match Command::new("ffmpeg").arg("-version").output() {
        Ok(_) => info!("ffmpeg found"),
        Err(_) => {
            tracing::error!(
                "ffmpeg not found! Install it: winget install ffmpeg  (Windows) / apt install ffmpeg (Linux)"
            );
            std::process::exit(1);
        }
    }

    info!("Loading whisper model: {}", model_path);
    let ctx = WhisperContext::new_with_params(&model_path, WhisperContextParameters::default())
        .context("failed to load whisper model")?;

    info!(
        "Ready. language={}, max_concurrent={}",
        language.as_deref().unwrap_or("auto"),
        max_concurrent
    );

    let state = Arc::new(AppState {
        whisper_ctx: ctx,
        language,
        semaphore: Arc::new(Semaphore::new(max_concurrent)),
    });

    let cors = CorsLayer::new()
        .allow_origin(Any)
        .allow_methods(Any)
        .allow_headers(Any);

    let app = Router::new()
        .route("/health", get(health))
        .route("/transcribe", post(transcribe))
        .layer(TraceLayer::new_for_http())
        .layer(cors)
        .with_state(state);

    info!("Listening on {bind_addr}");
    let listener = TcpListener::bind(&bind_addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
