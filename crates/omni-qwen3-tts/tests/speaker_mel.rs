//! CPU checks against the four official Base x-vector fixtures from
//! `docs/qwen3-tts-base.md`. Set `OMNI_QWEN3_TTS_BASE_FIXTURES` to the directory
//! containing the Safetensors, `OMNI_QWEN3_TTS_BASE_MODEL` to the Base checkpoint,
//! and `OMNI_QWEN3_TTS_MODEL` to the CustomVoice checkpoint.

use std::path::PathBuf;

use omni_qwen3_tts::config::{Config, TtsModelType};
use omni_qwen3_tts::mel;
use omni_qwen3_tts::weights::File;

#[test]
fn loads_base_and_custom_voice_configs() {
    if let Some(dir) = std::env::var_os("OMNI_QWEN3_TTS_BASE_MODEL") {
        let c = Config::load(&PathBuf::from(dir)).unwrap();
        let TtsModelType::Base { speaker_encoder_config } = c.model.tts_model_type else {
            panic!("expected Base checkpoint")
        };
        assert_eq!(speaker_encoder_config.sample_rate, 24_000);
        assert_eq!(speaker_encoder_config.mel_dim, 128);
        assert_eq!(speaker_encoder_config.enc_dim, 2048);
    } else {
        eprintln!("skipped Base config: set OMNI_QWEN3_TTS_BASE_MODEL");
    }
    if let Some(dir) = std::env::var_os("OMNI_QWEN3_TTS_MODEL") {
        let c = Config::load(&PathBuf::from(dir)).unwrap();
        assert!(matches!(c.model.tts_model_type, TtsModelType::CustomVoice));
        assert!(c.model.talker_config.spk_id.contains_key("ryan"));
    } else {
        eprintln!("skipped CustomVoice config: set OMNI_QWEN3_TTS_MODEL");
    }
}

#[test]
fn matches_official_speaker_mels() {
    let Some(dir) = std::env::var_os("OMNI_QWEN3_TTS_BASE_FIXTURES").map(PathBuf::from) else {
        eprintln!("skipped: set OMNI_QWEN3_TTS_BASE_FIXTURES");
        return;
    };
    for case in ["en-short", "en-long", "cn-short", "cn-long"] {
        let path = dir.join(format!("{case}-xvector-true.safetensors"));
        let fixture = File::open(&path).unwrap_or_else(|e| panic!("{}: {e:#}", path.display()));
        let pcm = fixture.host("ref_pcm").unwrap().data;
        let shape = [pcm.len() / 256, 128];
        let expected = fixture.expect("speaker_mel", &shape).unwrap();
        let expected_input = fixture.expect("speaker_mel_input", &shape).unwrap();

        let actual = mel::log_mel(&pcm).unwrap();
        assert_eq!(actual.len(), shape[0] * shape[1], "{case} log-mel size");
        assert!(actual.iter().all(|x| x.is_finite()), "{case} finite log-mel");
        let mut max_abs = 0.0;
        let mut squared = 0.0;
        let mut worst = 0;
        for (i, (a, b)) in actual.iter().zip(&expected.data).enumerate() {
            let error = (a - b).abs();
            if error > max_abs {
                max_abs = error;
                worst = i;
            }
            squared += f64::from(error).powi(2);
        }
        let rms = (squared / actual.len() as f64).sqrt();
        let expected_cast = mel::encoder_input(&expected.data);
        assert!(
            expected_cast.iter().zip(&expected_input.data).all(|(a, b)| a.to_f32() == *b),
            "{case} BF16 rounding of official mel differs from recorded input"
        );
        let actual_input = mel::encoder_input(&actual);
        let mut max_ulp = 0;
        let mut changed = 0;
        for (a, b) in actual_input.iter().zip(expected_cast) {
            let ulps = a.to_bits().abs_diff(b.to_bits());
            max_ulp = max_ulp.max(ulps);
            changed += usize::from(ulps != 0);
        }
        eprintln!(
            "{case}: {} frames, FP32 max abs {max_abs:.3e} (RMS {rms:.3e}), BF16 max ULP {max_ulp}, {changed}/{} changed",
            shape[0],
            actual.len()
        );
        eprintln!(
            "worst frame {}, band {}, actual {}, expected {}",
            worst / 128,
            worst % 128,
            actual[worst],
            expected.data[worst]
        );
        // The CPU FFT and Hann cosine round differently from PyTorch. The
        // largest log errors are in bins close to the 1e-5 magnitude floor.
        assert!(max_abs < 3e-3, "{case} FP32 max abs error {max_abs}");
        assert!(rms < 3e-5, "{case} FP32 RMS error {rms}");
        assert!(max_ulp <= 1, "{case} BF16 max ULP error {max_ulp}");
        assert!(changed * 1000 <= actual.len(), "{case} BF16 rounding differs at {changed} of {} values", actual.len());
    }
}
