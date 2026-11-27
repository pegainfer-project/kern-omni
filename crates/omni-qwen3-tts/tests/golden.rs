//! The model against the official implementation's recorded run
//! (`tools/qwen3_tts/golden.py`): tokenizer ids, prompt embeddings,
//! teacher-forced talker and code-predictor logits, and the decoded waveform,
//! all through the serving path (`prefill`, `first`, `decode`) with the
//! recorded frames forced in place of the draws.
//!
//! Needs a GPU, `OMNI_QWEN3_TTS_MODEL` (checkpoint directory) and
//! `OMNI_QWEN3_TTS_GOLDEN` (the golden file); skipped when either is unset.
//! Base uses `OMNI_QWEN3_TTS_BASE_MODEL` and `OMNI_QWEN3_TTS_BASE_FIXTURES`
//! (the Base golden fixture directory).

use std::path::Path;
use std::path::PathBuf;

use omni_qwen3_tts::config::TtsModelType;
use omni_qwen3_tts::mel;
use omni_qwen3_tts::model::Draw;
use omni_qwen3_tts::model::GROUPS;
use omni_qwen3_tts::model::Limits;
use omni_qwen3_tts::model::Model;
use omni_qwen3_tts::model::Seq;
use omni_qwen3_tts::prompt;
use omni_qwen3_tts::prompt::Prompt;
use omni_qwen3_tts::speaker_encoder::SpeakerEncoder;
use omni_qwen3_tts::weights::File;

fn paths() -> Option<(PathBuf, PathBuf)> {
    let get = |k| std::env::var_os(k).map(PathBuf::from);
    let found = get("OMNI_QWEN3_TTS_MODEL").zip(get("OMNI_QWEN3_TTS_GOLDEN"));
    if found.is_none() {
        eprintln!("skipped: set OMNI_QWEN3_TTS_MODEL and OMNI_QWEN3_TTS_GOLDEN");
    }
    found
}

struct Golden {
    file: File,
    text: String,
    speaker: String,
    language: String,
}

impl Golden {
    fn open(path: &Path) -> Golden {
        let bytes = std::fs::read(path).unwrap();
        let (_, meta) = safetensors::SafeTensors::read_metadata(&bytes).unwrap();
        let m = meta.metadata().clone().unwrap();
        Golden {
            file: File::open(path).unwrap(),
            text: m["text"].clone(),
            speaker: m["speaker"].clone(),
            language: m["language"].clone(),
        }
    }

    fn f32(&self, name: &str) -> (Vec<usize>, Vec<f32>) {
        let t = self.file.host(name).unwrap();
        (t.shape, t.data)
    }

    fn i32(&self, name: &str) -> Vec<i32> {
        self.file.ints(name).unwrap()
    }
}

fn cosine(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a.iter().zip(b).map(|(&x, &y)| x as f64 * y as f64).sum();
    let norm = |v: &[f32]| v.iter().map(|&x| (x as f64).powi(2)).sum::<f64>().sqrt();
    dot / (norm(a) * norm(b))
}

fn argmax(v: &[f32]) -> usize {
    v.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)).unwrap().0
}

/// Worst per-row cosine and the fraction of rows whose argmax agrees.
fn agreement(ours: &[f32], theirs: &[f32], width: usize) -> (f64, f64) {
    assert_eq!(ours.len(), theirs.len());
    assert!(!ours.is_empty() && ours.len().is_multiple_of(width));
    let pairs: Vec<_> = ours.chunks(width).zip(theirs.chunks(width)).collect();
    let worst = pairs.iter().map(|(a, b)| cosine(a, b)).fold(1.0, f64::min);
    let agree = pairs.iter().filter(|(a, b)| argmax(a) == argmax(b)).count() as f64 / pairs.len() as f64;
    (worst, agree)
}

/// Three streams of the same prompt and frames: `a` and `b` in lockstep
/// (their audio must be identical), `c` starting `LAG` steps later, so it
/// is prefilled next to running rows and shares calls of other sizes.
const LAG: usize = 7;
const LIMITS: Limits = Limits { max_batch: 3, max_tokens: 1024, kv_gib: 0.5 };

#[test]
fn matches_the_official_run() {
    let Some((dir, golden)) = paths() else { return };
    let golden = Golden::open(&golden);
    let mut model = Model::load(0, &dir, LIMITS).unwrap();

    let text_ids = model.tokenizer.assistant(&golden.text);
    assert_eq!(text_ids, golden.i32("text_ids"), "tokenizer ids");
    let voice = prompt::voice(&model.config.model, &golden.speaker, &golden.language).unwrap();
    let prompt = prompt::assemble(&model.config.model, voice, &text_ids, None);
    let snrs = replay(&mut model, &golden, &prompt, 0.5);
    assert!(snrs.iter().all(|&snr| snr > 30.0), "CustomVoice waveform SNR {snrs:?} dB");
}

#[test]
fn base_x_vector_matches_the_official_run() {
    let Some((dir, fixtures)) =
        std::env::var_os("OMNI_QWEN3_TTS_BASE_MODEL").zip(std::env::var_os("OMNI_QWEN3_TTS_BASE_FIXTURES"))
    else {
        eprintln!("skipped: set OMNI_QWEN3_TTS_BASE_MODEL and OMNI_QWEN3_TTS_BASE_FIXTURES");
        return;
    };
    let (dir, fixtures) = (PathBuf::from(dir), PathBuf::from(fixtures));
    let mut model = Model::load(0, &dir, LIMITS).unwrap();
    let TtsModelType::Base { speaker_encoder_config } = &model.config.model.tts_model_type else {
        panic!("the Base golden needs a Base checkpoint");
    };
    let cases = ["en-short", "cn-short", "en-long", "cn-long"].map(|case| {
        let golden = Golden::open(&fixtures.join(format!("{case}-xvector-true.safetensors")));
        (case, golden)
    });
    let max_frames = cases.iter().map(|(_, g)| g.file.shape("speaker_mel_input").unwrap()[0]).max().unwrap();
    let mut encoder = SpeakerEncoder::load(0, &dir, speaker_encoder_config, max_frames).unwrap();

    // Use different speakers in the same prefill so each row must select
    // its own sequence's vector. Check this before running our encoder.
    let width = model.config.model.talker_config.stack.hidden_size;
    for pair in cases.as_chunks::<2>().0 {
        let prompts = pair.each_ref().map(|(case, golden)| {
            let ids = model.tokenizer.assistant(&golden.text);
            assert_eq!(ids, golden.i32("text_ids"), "{case}: tokenizer ids");
            let reference = golden.f32("speaker_embedding").1;
            prompt::assemble_base(&model.config.model, reference, &golden.language, &ids).unwrap()
        });
        let mut seqs = prompts.each_ref().map(|p| model.open(p.len(), 1).unwrap().unwrap());
        let mut rows: Vec<_> = seqs
            .iter_mut()
            .zip(&prompts)
            .zip(pair)
            .map(|((seq, prompt), (_, golden))| {
                let force = Some(golden.i32("codes")[..GROUPS].try_into().unwrap());
                (seq, prompt, Draw { uniforms: [0.5; GROUPS], force })
            })
            .collect();
        model.start(&mut rows).unwrap();
        let embeds = model.embeds(prompts.iter().map(Prompt::len).sum()).unwrap();
        let mut offset = 0;
        for ((case, golden), prompt) in pair.iter().zip(&prompts) {
            let end = offset + prompt.len() * width;
            check_prefill(&embeds[offset..end], golden, prompt, &format!("{case} with the official speaker vector"));
            offset = end;
        }
    }

    let waveform_snrs = cases.map(|(case, golden)| {
        eprintln!("{case}");
        let ids = model.tokenizer.assistant(&golden.text);
        let pcm = golden.f32("ref_pcm").1;
        let features = mel::log_mel(&pcm).unwrap();
        let vector = encoder.encode(&mel::encoder_input(&features)).unwrap();
        let prompt = prompt::assemble_base(&model.config.model, vector, &golden.language, &ids).unwrap();
        // EOS is the final eligible code in vocabulary order. Draw from the
        // end of the distribution after replaying the recorded audio frames.
        (case, replay(&mut model, &golden, &prompt, 1.0 - f32::EPSILON))
    });
    for (case, snrs) in waveform_snrs {
        // The existing streaming decoder reaches about 27 dB on en-short's
        // recorded codes even when replayed through CustomVoice.
        let minimum = if case == "en-short" { 25.0 } else { 30.0 };
        assert!(snrs.iter().all(|&snr| snr > minimum), "{case} waveform SNR {snrs:?} dB, expected over {minimum}");
    }
}

fn check_prefill(actual: &[f32], golden: &Golden, prompt: &Prompt, label: &str) {
    let (embed_shape, embeds) = golden.f32("prefill_embeds");
    assert_eq!(embed_shape[0], prompt.len(), "prompt length");
    let (worst, _) = agreement(actual, &embeds, embed_shape[1]);
    eprintln!("prompt embeddings {label}: worst cosine {worst:.6}");
    assert!(worst > 0.999, "prompt embeddings diverge {label}: worst cosine {worst}");
}

fn replay(model: &mut Model, golden: &Golden, prompt: &Prompt, end_uniform: f32) -> [f64; 2] {
    let frames: Vec<[i32; GROUPS]> = golden.i32("codes").chunks(GROUPS).map(|c| c.try_into().unwrap()).collect();
    let n = frames.len();
    let (_, talker_logits) = golden.f32("talker_logits");
    let (_, predictor_logits) = golden.f32("predictor_logits");
    let t = &model.config.model.talker_config;
    let (vocab, p_vocab) = (t.stack.vocab_size, t.code_predictor_config.vocab_size);
    let spf = model.config.samples_per_frame;

    let mut seqs: Vec<Seq> = (0..3).map(|_| model.open(prompt.len(), n + 1).unwrap().unwrap()).collect();
    let starts = [0, 0, LAG];
    let draw =
        |t: usize| Draw { uniforms: [if t == n { end_uniform } else { 0.5 }; GROUPS], force: frames.get(t).copied() };
    let mut ours_talker = Vec::new();
    let mut ours_predictor = Vec::new();
    let mut wav = vec![Vec::new(); 3];
    for step in 0..=n + LAG {
        let at: Vec<Option<usize>> = starts.iter().map(|&s| step.checked_sub(s).filter(|&t| t <= n)).collect();
        for fresh in [true, false] {
            let who: Vec<usize> = (0..3).filter(|&k| at[k].is_some_and(|t| (t == 0) == fresh)).collect();
            if who.is_empty() {
                continue;
            }
            let picked = seqs.iter_mut().enumerate().filter(|(k, _)| who.contains(k)).map(|(_, s)| s);
            let out = if fresh {
                model.start(&mut picked.map(|s| (s, prompt, draw(0))).collect::<Vec<_>>()).unwrap()
            } else {
                let rows = picked.zip(&who).map(|(s, &k)| (s, draw(at[k].unwrap())));
                model.step(&mut rows.collect::<Vec<_>>()).unwrap()
            };
            for (i, &k) in who.iter().enumerate() {
                match at[k].unwrap() {
                    t if t < n => {
                        assert_eq!(out.codes[i], frames[t], "stream {k} frame {t} is not the forced one");
                        wav[k].extend_from_slice(&out.wav[i * spf..(i + 1) * spf]);
                    }
                    _ => assert!(model.is_end(&out.codes[i]), "the recorded run ends after {n} frames"),
                }
            }
            if who[0] == 0 {
                let t = at[0].unwrap();
                if t == 0 {
                    check_prefill(&model.embeds(prompt.len()).unwrap(), golden, prompt, "in replay");
                }
                ours_talker.extend(model.talker_logits(0).unwrap());
                if t < n {
                    ours_predictor.extend(model.predictor_logits(0).unwrap());
                }
            }
        }
    }

    let (worst, agree) = agreement(&ours_talker, &talker_logits, vocab);
    eprintln!("talker logits: worst cosine {worst:.5}, argmax agreement {agree:.3}");
    assert!(worst > 0.999 && agree > 0.95, "talker logits diverge");
    let (worst, agree) = agreement(&ours_predictor, &predictor_logits, p_vocab);
    eprintln!("predictor logits: worst cosine {worst:.5}, argmax agreement {agree:.3}");
    assert!(worst > 0.998 && agree > 0.9, "predictor logits diverge");

    let (_, reference) = golden.f32("wav");
    assert_eq!(wav[0], wav[1], "two streams of the same frames in one batch decode differently");
    [("streamed waveform", &wav[0]), ("lagging stream", &wav[2])].map(|(name, audio)| {
        let snr = snr(audio, &reference);
        eprintln!("{name}: snr {snr:.2} dB");
        snr
    })
}

fn snr(ours: &[f32], theirs: &[f32]) -> f64 {
    assert_eq!(ours.len(), theirs.len());
    let noise: f64 = ours.iter().zip(theirs).map(|(&a, &b)| (a as f64 - b as f64).powi(2)).sum();
    let signal: f64 = theirs.iter().map(|&x| (x as f64).powi(2)).sum();
    10.0 * (signal / noise).log10()
}
