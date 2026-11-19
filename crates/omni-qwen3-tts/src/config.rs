//! The checkpoint's JSON configs, reduced to the fields the engine reads.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::ensure;
use serde::Deserialize;
use serde::de::DeserializeOwned;

use crate::mel;

pub use omni_qwen3::config::Stack;

#[derive(Clone, Debug, Deserialize)]
pub struct Talker {
    #[serde(flatten)]
    pub stack: Stack,
    pub num_code_groups: usize,
    pub code_predictor_config: Stack,
    pub codec_bos_id: i32,
    pub codec_eos_token_id: i32,
    pub codec_pad_id: i32,
    pub codec_think_id: i32,
    pub codec_nothink_id: i32,
    pub codec_think_bos_id: i32,
    pub codec_think_eos_id: i32,
    pub codec_language_id: BTreeMap<String, i32>,
    pub spk_id: BTreeMap<String, i32>,
    /// `false`, or the dialect a speaker defaults to.
    pub spk_is_dialect: BTreeMap<String, serde_json::Value>,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Model {
    #[serde(flatten)]
    pub tts_model_type: TtsModelType,
    pub tts_bos_token_id: i32,
    pub tts_eos_token_id: i32,
    pub tts_pad_token_id: i32,
    pub talker_config: Talker,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(tag = "tts_model_type", rename_all = "snake_case")]
pub enum TtsModelType {
    Base { speaker_encoder_config: SpeakerEncoderConfig },
    CustomVoice,
}

#[derive(Clone, Debug, Deserialize)]
#[serde(default)]
pub struct SpeakerEncoderConfig {
    pub sample_rate: u32,
    pub mel_dim: usize,
    pub enc_dim: usize,
    pub enc_channels: Vec<usize>,
    pub enc_kernel_sizes: Vec<usize>,
    pub enc_dilations: Vec<usize>,
    pub enc_attention_channels: usize,
    pub enc_res2net_scale: usize,
    pub enc_se_channels: usize,
}

impl Default for SpeakerEncoderConfig {
    /// Base checkpoints can omit the speaker encoder's architectural defaults.
    fn default() -> Self {
        Self {
            sample_rate: mel::SAMPLE_RATE,
            mel_dim: mel::BINS,
            enc_dim: 1024,
            enc_channels: vec![512, 512, 512, 512, 1536],
            enc_kernel_sizes: vec![5, 3, 3, 3, 1],
            enc_dilations: vec![1, 2, 3, 4, 1],
            enc_attention_channels: 128,
            enc_res2net_scale: 8,
            enc_se_channels: 128,
        }
    }
}

#[derive(Clone, Copy, Debug, Deserialize)]
pub struct Generation {
    pub temperature: f32,
    pub top_k: i32,
    pub repetition_penalty: f32,
    pub subtalker_temperature: f32,
    pub subtalker_top_k: i32,
    pub max_new_tokens: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Codec {
    pub codebook_size: usize,
    pub codebook_dim: usize,
    pub latent_dim: usize,
    pub decoder_dim: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub rms_norm_eps: f32,
    pub rope_theta: f32,
    pub sliding_window: usize,
    pub num_quantizers: usize,
    pub upsample_rates: Vec<usize>,
    pub upsampling_ratios: Vec<usize>,
}

#[derive(Clone, Debug, Deserialize)]
struct CodecFile {
    output_sample_rate: u32,
    decode_upsample_rate: usize,
    decoder_config: Codec,
}

/// Everything read from a checkpoint directory's JSON.
#[derive(Clone, Debug)]
pub struct Config {
    pub model: Model,
    pub generation: Generation,
    pub codec: Codec,
    pub sample_rate: u32,
    /// PCM samples per codec frame.
    pub samples_per_frame: usize,
}

fn read<T: DeserializeOwned>(path: &Path) -> Result<T> {
    let text = std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_str(&text).with_context(|| format!("parsing {}", path.display()))
}

impl Config {
    pub fn load(dir: &Path) -> Result<Self> {
        let codec: CodecFile = read(&dir.join("speech_tokenizer/config.json"))?;
        let model: Model = read(&dir.join("config.json"))?;
        if let TtsModelType::Base { speaker_encoder_config } = &model.tts_model_type {
            let c = speaker_encoder_config;
            ensure!(
                c.sample_rate == mel::SAMPLE_RATE,
                "speaker encoder expects {} Hz, got {} Hz",
                mel::SAMPLE_RATE,
                c.sample_rate
            );
            ensure!(c.mel_dim == mel::BINS, "speaker encoder expects {} mel bins, got {}", mel::BINS, c.mel_dim);
            let hidden = model.talker_config.stack.hidden_size;
            ensure!(c.enc_dim == hidden, "talker expects speaker embeddings of width {hidden}, got {}", c.enc_dim);
        }
        Ok(Self {
            model,
            generation: read(&dir.join("generation_config.json"))?,
            sample_rate: codec.output_sample_rate,
            samples_per_frame: codec.decode_upsample_rate,
            codec: codec.decoder_config,
        })
    }
}
