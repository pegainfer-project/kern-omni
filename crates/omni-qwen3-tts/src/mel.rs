//! Qwen3-TTS Base's speaker-encoder frontend, matching the checkpoint's
//! periodic Hann STFT and Slaney-normalized magnitude mel filterbank.
//! Both the log-mel and its BF16 cast are in frame-major order.

use std::f32::consts::TAU;
use std::sync::Arc;
use std::sync::LazyLock;

use anyhow::{Result, ensure};
use half::bf16;
use realfft::{RealFftPlanner, RealToComplex};

pub const SAMPLE_RATE: u32 = 24_000;
pub const BINS: usize = 128;
const FFT: usize = 1024;
const HOP: usize = 256;
const PAD: usize = (FFT - HOP) / 2;

struct Band {
    start: usize,
    weights: Vec<f32>,
}

struct Frontend {
    fft: Arc<dyn RealToComplex<f32>>,
    window: [f32; FFT],
    bands: [Band; BINS],
}

static FRONTEND: LazyLock<Frontend> = LazyLock::new(|| Frontend {
    fft: RealFftPlanner::new().plan_fft_forward(FFT),
    window: std::array::from_fn(|i| 0.5 * (1.0 - (i as f32 * (TAU / FFT as f32)).cos())),
    bands: filterbank(),
});

fn filterbank() -> [Band; BINS] {
    // Slaney mel frequencies are linear until 1 kHz and logarithmic above it.
    const LINEAR_STEP: f64 = 200.0 / 3.0;
    const LOG_START_HZ: f64 = 1000.0;
    const LOG_START_MEL: f64 = LOG_START_HZ / LINEAR_STEP;
    let log_step = 6.4_f64.ln() / 27.0;
    let nyquist = f64::from(SAMPLE_RATE) / 2.0;
    let max_mel = LOG_START_MEL + (nyquist / LOG_START_HZ).ln() / log_step;
    let edges: [f64; BINS + 2] = std::array::from_fn(|i| {
        let mel = i as f64 * (max_mel / (BINS + 1) as f64);
        if mel < LOG_START_MEL { LINEAR_STEP * mel } else { LOG_START_HZ * (log_step * (mel - LOG_START_MEL)).exp() }
    });

    let bin_step = f64::from(SAMPLE_RATE) / FFT as f64;
    std::array::from_fn(|i| {
        let [left, center, right] = [edges[i], edges[i + 1], edges[i + 2]];
        let start = (left / bin_step).ceil() as usize;
        let end = (right / bin_step).ceil() as usize;
        let norm = 2.0 / (right - left);
        let weights = (start..end)
            .map(|bin| {
                let hz = bin as f64 * bin_step;
                let triangle = ((hz - left) / (center - left)).min((right - hz) / (right - center)).max(0.0) as f32;
                (f64::from(triangle) * norm) as f32
            })
            .collect();
        Band { start, weights }
    })
}

/// Convert checked mono 24 kHz PCM to FP32 log-mel, with `BINS` values per frame.
pub fn log_mel(pcm: &[f32]) -> Result<Vec<f32>> {
    let n = pcm.len();
    ensure!(n > PAD, "speaker mel needs at least {} PCM samples for reflection, got {n}", PAD + 1);

    // Reflection excludes the two endpoint samples. With center=false the
    // padded length gives exactly floor(n / HOP) STFT frames.
    let padded: Vec<f32> =
        pcm[1..=PAD].iter().rev().chain(pcm).chain(pcm[n - PAD - 1..n - 1].iter().rev()).copied().collect();
    let Frontend { fft, window, bands } = &*FRONTEND;
    let mut input = fft.make_input_vec();
    let mut spectrum = fft.make_output_vec();
    let mut scratch = fft.make_scratch_vec();
    let mut magnitude = [0.0; FFT / 2 + 1];
    let mut mel = Vec::with_capacity(n / HOP * BINS);
    for frame in padded.windows(FFT).step_by(HOP) {
        for (out, (&sample, &w)) in input.iter_mut().zip(frame.iter().zip(window)) {
            *out = sample * w;
        }
        fft.process_with_scratch(&mut input, &mut spectrum, &mut scratch)?;
        for (out, z) in magnitude.iter_mut().zip(&spectrum) {
            *out = (z.norm_sqr() + 1e-9).sqrt();
        }
        mel.extend(bands.iter().map(|Band { start, weights }| {
            let energy: f32 = weights.iter().zip(&magnitude[*start..]).map(|(w, mag)| w * mag).sum();
            energy.max(1e-5).ln()
        }));
    }
    Ok(mel)
}

/// Round the frame-major log-mel to the speaker encoder's input dtype.
pub fn encoder_input(mel: &[f32]) -> Vec<bf16> {
    mel.iter().copied().map(bf16::from_f32).collect()
}
