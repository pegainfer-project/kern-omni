//! ECAPA-TDNN against the official Base model on four recorded references.
//! Requires `OMNI_QWEN3_TTS_BASE_MODEL`, `OMNI_QWEN3_TTS_BASE_FIXTURES`
//! (the Base golden fixture directory) and a GPU compatible with `OMNI_CUDA_ARCH`.

use std::path::PathBuf;
use std::time::Instant;

use omni_qwen3_tts::config::{Config, TtsModelType};
use omni_qwen3_tts::mel;
use omni_qwen3_tts::speaker_encoder::SpeakerEncoder;
use omni_qwen3_tts::weights::File;

fn compare(
    case: &str,
    boundary: &str,
    frames: usize,
    input: &[half::bf16],
    model: &mut SpeakerEncoder,
    expected: &[f32],
) -> Vec<f32> {
    let start = Instant::now();
    let actual = model.encode(input).unwrap();
    let elapsed = start.elapsed();
    assert_eq!(actual.len(), expected.len(), "{case} {boundary} speaker embedding width");
    assert!(actual.iter().all(|v| v.is_finite()), "{case} {boundary} finite speaker embedding");
    let (mut dot, mut actual_sq, mut expected_sq, mut error_sq, mut max_error) = (0.0, 0.0, 0.0, 0.0, 0.0f64);
    for (&a, &b) in actual.iter().zip(expected) {
        let (a, b) = (f64::from(a), f64::from(b));
        dot += a * b;
        actual_sq += a * a;
        expected_sq += b * b;
        error_sq += (a - b).powi(2);
        max_error = max_error.max((a - b).abs());
    }
    let cosine = dot / (actual_sq * expected_sq).sqrt();
    let relative_l2 = (error_sq / expected_sq).sqrt();
    eprintln!(
        "{case} {boundary}: {frames} frames, cosine {cosine:.8}, rel-L2 {relative_l2:.6}, max abs {max_error:.6}, encoder {elapsed:?}"
    );
    assert!(
        cosine > 0.9999 && relative_l2 < 0.01 && max_error < 0.08,
        "{case} {boundary} speaker embedding differs from golden"
    );
    actual
}

#[test]
fn speaker_encoder_matches_recorded_base() {
    let get = |k| std::env::var_os(k).map(PathBuf::from);
    let Some((dir, fixtures)) = get("OMNI_QWEN3_TTS_BASE_MODEL").zip(get("OMNI_QWEN3_TTS_BASE_FIXTURES")) else {
        eprintln!("skipped: set OMNI_QWEN3_TTS_BASE_MODEL and OMNI_QWEN3_TTS_BASE_FIXTURES");
        return;
    };
    let config = Config::load(&dir).unwrap();
    let TtsModelType::Base { speaker_encoder_config: config } = &config.model.tts_model_type else {
        panic!("expected the Base checkpoint");
    };
    let cases = ["en-short", "cn-long", "cn-short", "en-long"]
        .map(|name| (name, File::open(&fixtures.join(format!("{name}-xvector-true.safetensors"))).unwrap()));
    let one_second = config.sample_rate as usize;
    let reference = cases[0].1.host("ref_pcm").unwrap().data;
    let longest_pcm: Vec<f32> = reference.iter().copied().cycle().take(30 * one_second).collect();
    let longest = mel::encoder_input(&mel::log_mel(&longest_pcm).unwrap());
    let max_frames = longest.len() / config.mel_dim;
    let mut encoder = SpeakerEncoder::load(0, &dir, config, max_frames).unwrap();
    let mut direct = Vec::new();
    for (case, file) in &cases {
        let mel = file.host("speaker_mel_input").unwrap();
        let expected = file.expect("speaker_embedding", &[config.enc_dim]).unwrap();
        assert_eq!(mel.shape, [mel.shape[0], config.mel_dim]);
        let input = mel::encoder_input(&mel.data);
        let actual = compare(case, "recorded mel", mel.shape[0], &input, &mut encoder, &expected.data);
        direct.push((input, actual));
    }
    for (case, file) in &cases {
        let pcm = file.host("ref_pcm").unwrap();
        let expected = file.expect("speaker_embedding", &[config.enc_dim]).unwrap();
        let mel = mel::log_mel(&pcm.data).unwrap();
        let input = mel::encoder_input(&mel);
        let frames = file.shape("speaker_mel_input").unwrap()[0];
        assert_eq!(input.len(), frames * config.mel_dim);
        compare(case, "computed mel", frames, &input, &mut encoder, &expected.data);
    }
    for (seconds, input) in [(1, mel::encoder_input(&mel::log_mel(&reference[..one_second]).unwrap())), (30, longest)] {
        let start = Instant::now();
        let result = encoder.encode(&input).unwrap();
        eprintln!("{seconds} s reference ({} frames), encoder {:?}", input.len() / config.mel_dim, start.elapsed());
        assert_eq!(result.len(), config.enc_dim);
        assert!(result.iter().all(|v| v.is_finite()));
    }
    for (input, original) in direct.into_iter().rev() {
        assert_eq!(encoder.encode(&input).unwrap(), original, "reused workspace changed the embedding");
    }
}
