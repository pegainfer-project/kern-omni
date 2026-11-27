//! Serving: requests in through an [`Inbox`], PCM chunks out through each
//! request's sink.
//!
//! One engine thread runs steps back to back. A step admits waiting requests
//! (FIFO, while batch slots, the per-step prompt budget and KV allow), starts
//! them (`prefill` and their first frame), advances every other running
//! request by one frame, and sends whichever requests have a chunk due. Each
//! call returns its frames already decoded to PCM (the codec streams, each
//! request keeping its decoder state), so a chunk's size decides only when
//! audio leaves, not what it costs.
//!
//! Base references wait outside admission: at most one checked PCM reference
//! and `max_batch` prepared prompts are held internally, with their combined
//! count also capped at `max_batch`. After decoding and emitting a tick, at
//! most one reference (30 seconds, 2,812 mel frames) is normalized, converted
//! to mel and speaker-encoded. The submission channel holds at most eight
//! more references, bounding queued PCM16 to 12.96 MB. No Talker state or KV
//! is leased until that preparation finishes.
//!
//! Admission leases a request's whole KV and its state slot up front, the KV
//! sized by its frame cap (proportional to the input length), so a running
//! request never waits for memory and nothing is ever preempted.
//!
//! The decisions are pure functions ([`chunk_due`], [`frame_cap`]);
//! [`Engine`] is the shell around them.

use std::collections::BTreeMap;
use std::collections::VecDeque;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::sync::mpsc::TryRecvError;
use std::thread::JoinHandle;
use std::time::Instant;

use anyhow::Context;
use anyhow::Result;
use bytes::Bytes;
use kern_pool::Denied;
use omni_engine::Done;
use omni_engine::EngineInfo;
use omni_engine::Event;
use omni_engine::Extra;
use omni_engine::Finish;
use omni_engine::Handle;
use omni_engine::Inbox;
use omni_engine::ReferenceAudioInfo;
use omni_engine::Speech;
use omni_engine::Submission;
use rand::RngExt;
use rand::SeedableRng;
use rand::rngs::StdRng;
use tokio::sync::mpsc::UnboundedSender;

use crate::config::TtsModelType;
use crate::mel;
use crate::model::Draw;
use crate::model::Limits;
use crate::model::Model;
use crate::model::Out;
use crate::model::Seq;
use crate::prompt;
use crate::prompt::Prompt;
use crate::speaker_encoder::SpeakerEncoder;

const REFERENCE_MAX_SECONDS: u32 = 30;

#[derive(Clone, Debug)]
pub struct Options {
    pub max_batch: usize,
    /// Prompt tokens per step.
    pub max_step_tokens: usize,
    pub kv_gib: f64,
    pub first_chunk_frames: usize,
    pub chunk_frames: usize,
}

impl Default for Options {
    fn default() -> Self {
        Self { max_batch: 64, max_step_tokens: 8192, kv_gib: 16.0, first_chunk_frames: 2, chunk_frames: 8 }
    }
}

/// Frames to decode now for a request with `generated` frames of which
/// `emitted` are already audio: the first chunk once `first` frames exist,
/// then every `steady`, and the remainder when it has `finished`.
pub fn chunk_due(generated: usize, emitted: usize, finished: bool, first: usize, steady: usize) -> Option<usize> {
    let pending = generated - emitted;
    let target = if emitted == 0 { first } else { steady };
    (pending > 0 && (pending >= target || finished)).then_some(pending.min(target))
}

/// The most frames a request may generate: generous for slow speech (Chinese
/// runs ~3 frames per character, English under 1), capped by the model.
pub fn frame_cap(input_chars: usize, model_max: usize) -> usize {
    (50 + 6 * input_chars).min(model_max)
}

/// The checkpoint's speakers or reference capability, languages and sampling seed.
pub fn info(model: &Model, name: &str, max_input_chars: usize) -> EngineInfo {
    let t = &model.config.model.talker_config;
    let languages = std::iter::once("auto".to_string()).chain(t.codec_language_id.keys().cloned()).collect();
    EngineInfo {
        model: name.into(),
        sample_rate: model.config.sample_rate,
        voices: match &model.config.model.tts_model_type {
            TtsModelType::CustomVoice => t.spk_id.keys().cloned().collect(),
            TtsModelType::Base { .. } => Default::default(),
        },
        extra: BTreeMap::from([
            ("language".to_string(), Extra::OneOf(languages)),
            ("seed".to_string(), Extra::Integer(0..=i64::MAX)),
        ]),
        speeds: 1.0..=1.0,
        max_input_chars,
        reference_audio: match &model.config.model.tts_model_type {
            TtsModelType::CustomVoice => None,
            TtsModelType::Base { speaker_encoder_config } => Some(ReferenceAudioInfo {
                sample_rate: speaker_encoder_config.sample_rate,
                duration_seconds: 1..=REFERENCE_MAX_SECONDS,
                audio_only: true,
                with_transcript: false,
            }),
        },
    }
}

struct Job {
    id: u64,
    sink: UnboundedSender<Event>,
    prompt: Prompt,
    seq: Option<Seq>,
    frames: usize,
    /// Decoded audio not yet sent, s16le.
    pcm: Vec<u8>,
    emitted: usize,
    rng: StdRng,
    cap: usize,
    ended: bool,
}

impl Job {
    fn done(&self) -> bool {
        self.ended || self.frames >= self.cap
    }

    fn draw(&mut self) -> Draw {
        Draw { uniforms: std::array::from_fn(|_| self.rng.random::<f32>()), force: None }
    }

    /// Takes a frame: the end, or its audio.
    fn take(&mut self, end: bool, wav: &[f32]) {
        if end {
            self.ended = true;
            return;
        }
        self.frames += 1;
        self.pcm.extend(wav.iter().flat_map(|&x| ((x * 32767.0).round() as i16).to_le_bytes()));
    }
}

pub struct Engine {
    model: Model,
    opts: Options,
    speaker_encoder: Option<SpeakerEncoder>,
    pending: Option<Submission>,
    waiting: VecDeque<Job>,
    running: Vec<Job>,
}

/// Hands each job its row of `out`.
fn take(jobs: &mut [Job], out: &Out, model: &Model, spf: usize) {
    let wav = out.wav.chunks(spf);
    jobs.iter_mut().zip(&out.codes).zip(wav).for_each(|((j, codes), wav)| j.take(model.is_end(codes), wav));
}

fn abort(sink: &UnboundedSender<Event>) {
    let _ = sink.send(Event::Done(Done { finish: Finish::Aborted, input_units: 0, frames: 0 }));
}

impl Engine {
    pub fn new(model: Model, opts: Options) -> Self {
        Self { model, opts, speaker_encoder: None, pending: None, waiting: VecDeque::new(), running: Vec::new() }
    }

    /// A Base engine owns a standalone encoder on the same GPU thread as its model.
    pub fn with_speaker_encoder(model: Model, opts: Options, speaker_encoder: SpeakerEncoder) -> Self {
        Self { speaker_encoder: Some(speaker_encoder), ..Self::new(model, opts) }
    }

    fn job(&self, speech: &Speech, speaker: Option<Vec<f32>>, sink: UnboundedSender<Event>) -> Result<Job, String> {
        let m = &self.model;
        let language = speech.extra.get("language").and_then(|v| v.as_str()).unwrap_or("auto");
        let text = m.tokenizer.assistant(&speech.input);
        let prompt = if let Some(speaker) = speaker {
            prompt::assemble_base(&m.config.model, speaker, language, &text).map_err(|e| e.to_string())?
        } else {
            let voice = prompt::voice(&m.config.model, &speech.voice, language)?;
            let instruct = speech.instructions.as_deref().filter(|s| !s.is_empty()).map(|s| m.tokenizer.instruct(s));
            prompt::assemble(&m.config.model, voice, &text, instruct.as_deref())
        };
        if prompt.len() > self.opts.max_step_tokens {
            return Err(format!("{} prompt tokens exceed the step's {}", prompt.len(), self.opts.max_step_tokens));
        }
        let seed = speech.extra.get("seed").and_then(|v| v.as_u64()).unwrap_or_else(rand::random);
        Ok(Job {
            id: speech.id,
            sink,
            prompt,
            seq: None,
            frames: 0,
            pcm: vec![],
            emitted: 0,
            rng: StdRng::seed_from_u64(seed),
            cap: frame_cap(speech.input.chars().count(), m.config.generation.max_new_tokens),
            ended: false,
        })
    }

    fn enqueue(&mut self, speech: Speech, sink: UnboundedSender<Event>) {
        if sink.is_closed() {
            return;
        }
        if speech.reference.is_some() {
            debug_assert!(self.pending.is_none());
            self.pending = Some(Submission { speech, sink });
            return;
        }
        self.enqueue_prepared(speech, None, sink);
    }

    fn enqueue_prepared(&mut self, speech: Speech, speaker: Option<Vec<f32>>, sink: UnboundedSender<Event>) {
        if sink.is_closed() {
            return;
        }
        match self.job(&speech, speaker, sink.clone()) {
            Ok(job) => self.waiting.push_back(job),
            Err(e) => {
                tracing::warn!(id = speech.id, "rejected: {e}");
                abort(&sink);
            }
        }
    }

    fn waiting(&self) -> usize {
        self.waiting.len() + usize::from(self.pending.is_some())
    }

    fn can_receive(&self) -> bool {
        self.pending.is_none() && self.waiting() < self.opts.max_batch.max(1)
    }

    fn cancel(&mut self) {
        if self.pending.as_ref().is_some_and(|s| s.sink.is_closed()) {
            self.pending = None;
        }
        self.waiting.retain(|j| !j.sink.is_closed());
        self.running.retain(|j| !j.sink.is_closed());
    }

    /// The full reference is one ECAPA input: its gating and attentive pooling
    /// use global statistics, so splitting it into independent chunks is invalid.
    fn prepare(&mut self) -> Result<()> {
        let Some(Submission { mut speech, sink }) = self.pending.take() else {
            return Ok(());
        };
        if sink.is_closed() {
            return Ok(());
        }
        let start = Instant::now();
        let reference = speech.reference.take().expect("checked reference request");
        let pcm: Vec<f32> =
            reference.pcm.as_chunks::<2>().0.iter().map(|&s| f32::from(i16::from_le_bytes(s)) / 32768.0).collect();
        drop(reference);
        let pcm_ms = start.elapsed().as_secs_f64() * 1000.0;
        if sink.is_closed() {
            return Ok(());
        }
        let samples = pcm.len();
        let mel = mel::encoder_input(&mel::log_mel(&pcm)?);
        drop(pcm);
        let mel_ms = start.elapsed().as_secs_f64() * 1000.0 - pcm_ms;
        if sink.is_closed() {
            return Ok(());
        }
        let encoding = Instant::now();
        let speaker = self.speaker_encoder.as_mut().context("Base engine requires a speaker encoder")?.encode(&mel)?;
        tracing::debug!(
            id = speech.id,
            samples,
            frames = mel.len() / mel::BINS,
            pcm_ms,
            mel_ms,
            encoder_ms = encoding.elapsed().as_secs_f64() * 1000.0,
            prepare_ms = start.elapsed().as_secs_f64() * 1000.0,
            "prepared reference"
        );
        self.enqueue_prepared(speech, Some(speaker), sink);
        Ok(())
    }

    /// Leases waiting requests in order; returns how many joined `running`.
    fn admit(&mut self) -> Result<usize> {
        let mut tokens = 0;
        let mut admitted = 0;
        while let Some(job) = self.waiting.front() {
            if job.sink.is_closed() {
                self.waiting.pop_front();
                continue;
            }
            if self.running.len() >= self.opts.max_batch || tokens + job.prompt.len() > self.opts.max_step_tokens {
                break;
            }
            match self.model.open(job.prompt.len(), job.cap)? {
                Ok(seq) => {
                    let mut job = self.waiting.pop_front().expect("front exists");
                    tokens += job.prompt.len();
                    job.seq = Some(seq);
                    self.running.push(job);
                    admitted += 1;
                }
                Err(Denied::Busy | Denied::Remapping) => break,
                Err(d) => {
                    let job = self.waiting.pop_front().expect("front exists");
                    tracing::warn!(
                        id = job.id,
                        "rejected: {} prompt tokens and {} frames: {d}",
                        job.prompt.len(),
                        job.cap
                    );
                    abort(&job.sink);
                }
            }
        }
        Ok(admitted)
    }

    /// One step: admit and start, one frame for every other running request, due chunks out.
    pub fn step(&mut self) -> Result<()> {
        self.cancel();
        let fresh = self.admit()?;
        let spf = self.model.config.samples_per_frame;
        let split = self.running.len() - fresh;
        let (old, new) = self.running.split_at_mut(split);
        if !new.is_empty() {
            let draws: Vec<Draw> = new.iter_mut().map(Job::draw).collect();
            let mut rows: Vec<_> =
                new.iter_mut().zip(draws).map(|(j, d)| (j.seq.as_mut().expect("admitted"), &j.prompt, d)).collect();
            let out = self.model.start(&mut rows)?;
            take(new, &out, &self.model, spf);
        }
        if !old.is_empty() {
            let draws: Vec<Draw> = old.iter_mut().map(Job::draw).collect();
            let mut rows: Vec<_> =
                old.iter_mut().zip(draws).map(|(j, d)| (j.seq.as_mut().expect("admitted"), d)).collect();
            let out = self.model.step(&mut rows)?;
            take(old, &out, &self.model, spf);
        }
        self.emit();
        self.prepare()?;
        Ok(())
    }

    fn emit(&mut self) {
        let (first, steady, spf) =
            (self.opts.first_chunk_frames, self.opts.chunk_frames, self.model.config.samples_per_frame);
        self.running.retain_mut(|j| {
            while let Some(n) = chunk_due(j.frames, j.emitted, j.done(), first, steady) {
                let rest = j.pcm.split_off(n * spf * 2);
                let pcm = std::mem::replace(&mut j.pcm, rest);
                j.emitted += n;
                if j.sink.send(Event::Audio(Bytes::from(pcm))).is_err() {
                    tracing::debug!(id = j.id, "client gone");
                    return false;
                }
            }
            if j.done() {
                let done =
                    Done { finish: Finish::Complete, input_units: j.prompt.len() as u32, frames: j.frames as u32 };
                let _ = j.sink.send(Event::Done(done));
                return false;
            }
            true
        });
    }

    fn idle(&self) -> bool {
        self.pending.is_none() && self.waiting.is_empty() && self.running.is_empty()
    }
}

/// Loads the checkpoint at `dir` on a thread of its own and serves from it
/// until every [`omni_engine::Handle`] is dropped and the admitted work is
/// done. The model is loaded where it runs: its kern runtime is bound to the
/// thread that made it.
pub fn start(
    device: usize,
    dir: PathBuf,
    opts: Options,
    name: String,
    max_input_chars: usize,
    queue: usize,
) -> Result<(Handle, JoinHandle<()>)> {
    let (tx, rx) = std::sync::mpsc::sync_channel(1);
    let thread = std::thread::Builder::new().name("omni-qwen3-tts".into()).spawn(move || {
        let loaded = (|| {
            let limits = Limits { max_batch: opts.max_batch, max_tokens: opts.max_step_tokens, kv_gib: opts.kv_gib };
            let model = Model::load(device, &dir, limits).with_context(|| format!("load {}", dir.display()))?;
            let info = info(&model, &name, max_input_chars);
            let queue = if info.reference_audio.is_some() { queue.min(8) } else { queue };
            let (handle, inbox) = omni_engine::channel(info, queue);
            let encoder = match &model.config.model.tts_model_type {
                TtsModelType::CustomVoice => None,
                TtsModelType::Base { speaker_encoder_config } => {
                    // Warm the maximum shape before readiness, including the CPU
                    // FFT/filterbank setup and the encoder's GEMM algorithms.
                    let pcm = vec![0.0; speaker_encoder_config.sample_rate as usize * REFERENCE_MAX_SECONDS as usize];
                    let mel = mel::encoder_input(&mel::log_mel(&pcm)?);
                    let mut encoder =
                        SpeakerEncoder::load(device, &dir, speaker_encoder_config, mel.len() / mel::BINS)?;
                    encoder.encode(&mel)?;
                    Some(encoder)
                }
            };
            let engine = match encoder {
                Some(encoder) => Engine::with_speaker_encoder(model, opts, encoder),
                None => Engine::new(model, opts),
            };
            anyhow::Ok((handle, inbox, engine))
        })();
        match loaded {
            Ok((handle, inbox, engine)) => {
                let _ = tx.send(Ok(handle));
                if let Err(e) = run(inbox, engine) {
                    tracing::error!("engine stopped: {e:#}");
                }
            }
            Err(e) => {
                let _ = tx.send(Err(e));
            }
        }
    })?;
    let handle = rx.recv().context("the engine thread died while loading")??;
    Ok((handle, thread))
}

fn run(inbox: Inbox, mut engine: Engine) -> Result<()> {
    let mut open = true;
    loop {
        engine.cancel();
        if engine.idle() {
            inbox.load.waiting.store(0, Ordering::Relaxed);
            inbox.load.running.store(0, Ordering::Relaxed);
            if !open {
                return Ok(());
            }
            match inbox.rx.recv() {
                Ok(s) => engine.enqueue(s.speech, s.sink),
                Err(_) => return Ok(()),
            }
        }
        for _ in 0..engine.opts.max_batch.max(1) {
            if !engine.can_receive() {
                break;
            }
            match inbox.rx.try_recv() {
                Ok(s) => engine.enqueue(s.speech, s.sink),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    open = false;
                    break;
                }
            }
        }
        inbox.load.waiting.store(engine.waiting(), Ordering::Relaxed);
        inbox.load.running.store(engine.running.len(), Ordering::Relaxed);
        engine.step()?;
        inbox.load.waiting.store(engine.waiting(), Ordering::Relaxed);
        inbox.load.running.store(engine.running.len(), Ordering::Relaxed);
    }
}
