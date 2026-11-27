//! The talker's prompt: text tokens and the dual-track sequence built from them.
//!
//! Every prompt position carries a text-track token, projected through the
//! talker's text MLP, and optionally a codec-track embedding added on top.
//! The layout is the official non-streaming one (the whole text is known up front):
//!
//! ```text
//! [instruct…]  <|im_start|>assistant\n  pad…pad bos   text… eos   pad
//!  -            -                        think… spk    pad… pad    bos
//! ```
//!
//! The speaker position holds a codec token for CustomVoice and a continuous
//! x-vector for Base. After the prompt, every decode step's text track is
//! `<tts_pad>`.

use std::path::Path;

use anyhow::Context;
use anyhow::Result;
use anyhow::bail;
use anyhow::ensure;
use serde_json::Value;
use serde_json::json;

use crate::config::Model;
use crate::config::TtsModelType;

/// No codec-track token at this position.
pub const NO_CODEC: i32 = -1;

/// The contribution added to a row's projected text. Reference codec frames
/// can be represented here by their full set of codebook IDs when ICL is added.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum Codec {
    None,
    Token(i32),
    Speaker(Vec<f32>),
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Prompt {
    pub(crate) text: Vec<i32>,
    pub(crate) codec: Vec<Codec>,
}

impl Prompt {
    pub fn len(&self) -> usize {
        self.text.len()
    }

    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    fn push(&mut self, text: i32, codec: Codec) {
        self.text.push(text);
        self.codec.push(codec);
    }
}

/// A speaker's codec token and the language tag it speaks under.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Voice {
    pub speaker: i32,
    pub language: Option<i32>,
}

/// Resolves a speaker and a language name (`auto` lets the model infer it). A
/// dialect speaker asked for Chinese or `auto` speaks its dialect.
pub fn voice(model: &Model, speaker: &str, language: &str) -> Result<Voice, String> {
    let t = &model.talker_config;
    let speaker_id = *t.spk_id.get(speaker).ok_or_else(|| format!("unknown speaker `{speaker}`"))?;
    let dialect = t.spk_is_dialect.get(speaker).and_then(Value::as_str);
    let language = match (language, dialect) {
        ("auto" | "chinese", Some(d)) => Some(d),
        ("auto", None) => None,
        (l, _) => Some(l),
    };
    Ok(Voice { speaker: speaker_id, language: language_id(model, language)? })
}

fn language_id(model: &Model, language: Option<&str>) -> Result<Option<i32>, String> {
    language
        .map(|l| model.talker_config.codec_language_id.get(l).copied().ok_or_else(|| format!("unknown language `{l}`")))
        .transpose()
}

/// Lays out the prompt for `text` (the template-tokenized text, see
/// [`Tokenizer::assistant`]) and an optional instruction (see
/// [`Tokenizer::instruct`]).
pub fn assemble(model: &Model, voice: Voice, text: &[i32], instruct: Option<&[i32]>) -> Prompt {
    assemble_with_speaker(model, Codec::Token(voice.speaker), voice.language, text, instruct)
}

/// Assembles Base's x-vector-only prompt for template-tokenized target text
/// (see [`Tokenizer::assistant`]) and its target language (`auto` or a model
/// language name). The speaker vector is added at the usual speaker position.
pub fn assemble_base(model: &Model, speaker_embedding: Vec<f32>, language: &str, text: &[i32]) -> Result<Prompt> {
    ensure!(matches!(model.tts_model_type, TtsModelType::Base { .. }), "x-vector prompt requires a Base checkpoint");
    let hidden = model.talker_config.stack.hidden_size;
    ensure!(speaker_embedding.len() == hidden, "expected {hidden} speaker values, got {}", speaker_embedding.len());
    ensure!(speaker_embedding.iter().all(|x| x.is_finite()), "speaker values must be finite");
    let language = language.to_ascii_lowercase();
    let language = language_id(model, (language != "auto").then_some(language.as_str())).map_err(anyhow::Error::msg)?;
    Ok(assemble_with_speaker(model, Codec::Speaker(speaker_embedding), language, text, None))
}

fn assemble_with_speaker(
    model: &Model,
    speaker: Codec,
    language: Option<i32>,
    text: &[i32],
    instruct: Option<&[i32]>,
) -> Prompt {
    let t = &model.talker_config;
    let (bos, eos, pad) = (model.tts_bos_token_id, model.tts_eos_token_id, model.tts_pad_token_id);
    let mut p = Prompt::default();
    instruct.unwrap_or_default().iter().chain(&text[..3]).for_each(|&id| p.push(id, Codec::None));
    p.push(pad, Codec::Token(if language.is_some() { t.codec_think_id } else { t.codec_nothink_id }));
    p.push(pad, Codec::Token(t.codec_think_bos_id));
    if let Some(id) = language {
        p.push(pad, Codec::Token(id));
    }
    p.push(pad, Codec::Token(t.codec_think_eos_id));
    p.push(pad, speaker);
    p.push(bos, Codec::Token(t.codec_pad_id));
    text[3..text.len() - 5].iter().for_each(|&id| p.push(id, Codec::Token(t.codec_pad_id)));
    p.push(eos, Codec::Token(t.codec_pad_id));
    p.push(pad, Codec::Token(t.codec_bos_id));
    p
}

/// The Qwen2 byte-level BPE the checkpoint ships as `vocab.json` + `merges.txt`.
pub struct Tokenizer(tokenizers::Tokenizer);

impl Tokenizer {
    /// Builds the tokenizer the way `transformers` converts `Qwen2Tokenizer`:
    /// NFC, the Qwen2 split regex, byte-level BPE, and the special tokens of
    /// `tokenizer_config.json` at their declared ids.
    pub fn load(dir: &Path) -> Result<Self> {
        let read = |name: &str| std::fs::read_to_string(dir.join(name)).with_context(|| format!("reading {name}"));
        let vocab: Value = serde_json::from_str(&read("vocab.json")?)?;
        let merges: Vec<String> =
            read("merges.txt")?.lines().filter(|l| !l.starts_with("#version")).map(str::to_owned).collect();
        let config: Value = serde_json::from_str(&read("tokenizer_config.json")?)?;
        let added: Vec<Value> = config["added_tokens_decoder"]
            .as_object()
            .context("tokenizer_config.json has no added_tokens_decoder")?
            .iter()
            .map(|(id, t)| {
                json!({ "id": id.parse::<u64>().unwrap_or(u64::MAX), "content": t["content"], "single_word": false,
                        "lstrip": false, "rstrip": false, "normalized": false, "special": t["special"] })
            })
            .collect();
        let byte_level =
            json!({ "type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": false });
        let spec = json!({
            "version": "1.0",
            "added_tokens": added,
            "normalizer": { "type": "NFC" },
            "pre_tokenizer": { "type": "Sequence", "pretokenizers": [
                { "type": "Split", "behavior": "Isolated", "invert": false, "pattern": { "Regex":
                  "(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\\r\\n\\p{L}\\p{N}]?\\p{L}+|\\p{N}| ?[^\\s\\p{L}\\p{N}]+[\\r\\n]*|\\s*[\\r\\n]+|\\s+(?!\\S)|\\s+" } },
                byte_level,
            ] },
            "post_processor": null,
            "decoder": byte_level,
            "model": { "type": "BPE", "dropout": null, "unk_token": null, "continuing_subword_prefix": "",
                       "end_of_word_suffix": "", "fuse_unk": false, "byte_fallback": false, "ignore_merges": false,
                       "vocab": vocab, "merges": merges },
        });
        let inner: tokenizers::Tokenizer = spec.to_string().parse().map_err(|e| anyhow::anyhow!("tokenizer: {e}"))?;
        for t in config["added_tokens_decoder"].as_object().into_iter().flatten() {
            let content = t.1["content"].as_str().unwrap_or_default();
            if inner.token_to_id(content).map(|i| i.to_string()).as_deref() != Some(t.0.as_str()) {
                bail!("special token {content} did not land at id {}", t.0);
            }
        }
        Ok(Self(inner))
    }

    fn encode(&self, text: &str) -> Vec<i32> {
        let encoding = self.0.encode(text, false).expect("byte-level BPE encodes any string");
        encoding.get_ids().iter().map(|&i| i as i32).collect()
    }

    /// `<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n`.
    pub fn assistant(&self, text: &str) -> Vec<i32> {
        self.encode(&format!("<|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n"))
    }

    /// `<|im_start|>user\n{instruct}<|im_end|>\n`.
    pub fn instruct(&self, instruct: &str) -> Vec<i32> {
        self.encode(&format!("<|im_start|>user\n{instruct}<|im_end|>\n"))
    }
}
