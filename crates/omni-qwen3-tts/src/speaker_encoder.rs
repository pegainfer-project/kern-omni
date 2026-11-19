//! The Base checkpoint's ECAPA-TDNN as a standalone, eager kern program.
//! The mel input and every temporal workspace use frame-major BF16. A call
//! owns the whole reference: reflection, channel gating and attentive pooling
//! depend on its actual length. Workspaces are sized once at the frame cap.

use std::collections::BTreeMap;
use std::path::Path;

use anyhow::Result;
use anyhow::ensure;
use half::bf16;
use kern_manifest::Verified;
use kern_runtime::Runtime;
use omni_kern::Gen;
use omni_kern::HostTensors;
use omni_kern::THREADS;
use omni_kern::bf16s;
use omni_kern::buf;
use omni_kern::count;
use omni_kern::hex;
use omni_kern::i32a;
use omni_kern::inb;
use omni_kern::io;
use omni_kern::kernels_dir;
use omni_kern::outb;
use omni_kern::weights::File;
use omni_kern::weights::conv_taps;
use serde_json::Value;
use serde_json::json;
use sha2::Digest;

use crate::config::SpeakerEncoderConfig;

const CUBIN: &[u8] = include_bytes!(concat!(env!("OUT_DIR"), "/ecapa.cubin"));

pub struct SpeakerEncoder {
    runtime: Runtime,
    mel_dim: usize,
    min_frames: usize,
    max_frames: usize,
    upload: Vec<u8>,
}

impl SpeakerEncoder {
    /// Load only the speaker encoder's weights and scratch, before leasing
    /// any Talker sequence state. `max_frames` caps the mel input's time axis.
    pub fn load(device: usize, dir: &Path, config: &SpeakerEncoderConfig, max_frames: usize) -> Result<Self> {
        let min_frames = check_layout(config)?;
        ensure!(
            max_frames >= min_frames,
            "speaker encoder needs at least {min_frames} frames, frame cap is {max_frames}"
        );
        let sha = hex(&sha2::Sha256::digest(CUBIN));
        let checkpoint = File::open(&dir.join("model.safetensors"))?;
        let (manifest, tensors) = program(&checkpoint, config, max_frames, &sha)?;
        let verified = Verified::from_json(&manifest.to_string())?;
        let kernels = kernels_dir(&[("ecapa", &sha, CUBIN)])?;
        let mut runtime = Runtime::load(&verified, Some(&kernels), device, None, None)?;
        runtime.load_weights(&tensors)?;
        Ok(Self { runtime, mel_dim: config.mel_dim, min_frames, max_frames, upload: Vec::new() })
    }

    /// Encode `[frames, mel_dim]` into the checkpoint's BF16 speaker vector,
    /// widened to f32 for use in a prompt.
    pub fn encode(&mut self, mel: &[bf16]) -> Result<Vec<f32>> {
        ensure!(mel.len().is_multiple_of(self.mel_dim), "speaker encoder expects {} mel bins per frame", self.mel_dim);
        let frames = mel.len() / self.mel_dim;
        ensure!(
            (self.min_frames..=self.max_frames).contains(&frames),
            "speaker encoder expects {}–{} frames, got {frames}",
            self.min_frames,
            self.max_frames
        );
        let vars = BTreeMap::from([("frames".into(), frames as u64)]);
        self.upload.clear();
        self.upload.extend(mel.iter().flat_map(|b| b.to_le_bytes()));
        self.runtime.write_input_at("mel", &self.upload, &vars)?;
        self.runtime.issue("speaker_encoder", &vars)?;
        Ok(bf16s(&self.runtime.read_output("speaker")?))
    }
}

/// The residual blocks preserve their channel width, and MFA joins them all.
/// Constant-width residual stages also share the same temporary storage.
fn check_layout(c: &SpeakerEncoderConfig) -> Result<usize> {
    let channels = &c.enc_channels;
    let n = channels.len();
    ensure!(
        n >= 3 && c.enc_kernel_sizes.len() == n && c.enc_dilations.len() == n,
        "speaker encoder needs matching channels, kernel sizes and dilations for at least three layers"
    );
    let width = channels[0];
    ensure!(
        c.enc_res2net_scale >= 2 && width > 0 && width.is_multiple_of(c.enc_res2net_scale),
        "speaker encoder width {width} must split into {} Res2Net groups",
        c.enc_res2net_scale
    );
    ensure!(
        channels[..n - 1].iter().all(|&w| w == width) && channels[n - 1] == width * (n - 2),
        "speaker encoder needs equal residual widths and one MFA channel per residual output"
    );
    ensure!(
        [c.mel_dim, c.enc_dim, c.enc_se_channels, c.enc_attention_channels].into_iter().all(|w| w > 0),
        "speaker encoder channel widths must be positive"
    );
    ensure!(
        c.enc_kernel_sizes.iter().all(|&k| k % 2 == 1) && c.enc_dilations.iter().all(|&d| d > 0),
        "speaker encoder needs odd convolution kernels and positive dilations"
    );
    // Reflect padding is defined only when each side is shorter than the input.
    Ok(1 + c.enc_kernel_sizes.iter().zip(&c.enc_dilations).map(|(k, d)| k / 2 * d).max().unwrap())
}

#[derive(Clone, Copy)]
struct Signal {
    name: &'static str,
    width: usize,
    stride: usize,
    start: usize,
}

impl Signal {
    fn new(name: &'static str, width: usize) -> Self {
        Self { name, width, stride: width, start: 0 }
    }

    fn group(self, width: usize, group: usize) -> Self {
        Self { width, start: group * width, ..self }
    }
}

#[derive(Clone, Copy)]
enum Rows {
    Frames,
    One,
}

impl Rows {
    fn arg(self) -> (&'static str, Value) {
        match self {
            Self::Frames => ("i32", count(&json!("frames"))),
            Self::One => i32a(1),
        }
    }

    fn grid(self, width: usize) -> [Value; 3] {
        let blocks = match self {
            Self::Frames => json!({"ceil_div": [{"mul": ["frames", width]}, THREADS]}),
            Self::One => json!(width.div_ceil(THREADS as usize)),
        };
        [blocks, json!(1), json!(1)]
    }
}

#[derive(Clone, Copy)]
struct Window {
    rows: Rows,
    taps: usize,
    dilation: usize,
    previous_group: Option<usize>,
}

const POINT: Window = Window { rows: Rows::Frames, taps: 1, dilation: 1, previous_group: None };
const POOLED: Window = Window { rows: Rows::One, ..POINT };

#[derive(Clone, Copy)]
enum Activation {
    Identity,
    Relu,
    ReluTanh,
    Sigmoid,
}

struct Program<'a> {
    g: Gen,
    checkpoint: &'a File,
    columns: usize,
    product: usize,
}

/// The extra column folds bias into the FP32 accumulation before the one
/// BF16 output rounding. The rest is alignment padding for cuBLASLt.
fn packed(width: usize) -> usize {
    (width + 1).next_multiple_of(16)
}

impl Program<'_> {
    fn columns(&mut self, label: &str, x: Signal, window: Window) {
        let Window { rows, taps, dilation, previous_group } = window;
        let width = packed(x.width * taps);
        self.columns = self.columns.max(width);
        let (previous, start) = match previous_group {
            Some(start) => ("res2", start as i32),
            None => (x.name, -1),
        };
        self.g.launch(
            label,
            "ecapa_columns",
            rows.grid(width),
            THREADS,
            vec![
                inb(x.name),
                inb(previous),
                outb("columns"),
                rows.arg(),
                i32a(x.stride),
                i32a(x.start),
                i32a(x.width),
                i32a(start),
                i32a(taps),
                i32a(dilation),
                i32a(width),
            ],
        );
    }

    fn affine(&mut self, name: &str, input: usize, taps: usize, y: Signal, rows: Rows) -> Result<()> {
        let prefix = format!("speaker_encoder.{name}");
        let weight = self.checkpoint.expect(&format!("{prefix}.weight"), &[y.width, input, taps])?;
        let bias = self.checkpoint.expect(&format!("{prefix}.bias"), &[y.width])?;
        let weights = if taps == 1 { weight.data } else { conv_taps(&weight) };
        let inputs = input * taps;
        let width = packed(inputs);
        let mut matrix = vec![0.0; y.width * width];
        for ((dst, values), bias) in matrix.chunks_exact_mut(width).zip(weights.chunks_exact(inputs)).zip(bias.data) {
            dst[..inputs].copy_from_slice(values);
            dst[inputs] = bias;
        }
        let weight = self.g.weight(&prefix, &[y.width, width], &matrix);
        self.g.gemm_rows(name, (buf(y.name), buf("columns"), buf(&weight)), rows.arg().1, (y.width, width));
        Ok(())
    }

    fn place(&mut self, name: &str, x: Signal, y: Signal, rows: Rows, activation: Activation) {
        let out = if y.start > 0 { io(y.name) } else { outb(y.name) };
        self.g.launch(
            name,
            "ecapa_place",
            rows.grid(y.width),
            THREADS,
            vec![
                inb(x.name),
                out,
                rows.arg(),
                i32a(x.stride),
                i32a(x.start),
                i32a(y.stride),
                i32a(y.start),
                i32a(y.width),
                i32a(activation as i32),
            ],
        );
    }

    fn project(
        &mut self,
        name: &str,
        input: usize,
        taps: usize,
        y: Signal,
        rows: Rows,
        activation: Activation,
    ) -> Result<()> {
        self.product = self.product.max(y.width);
        let product = Signal::new("product", y.width);
        self.affine(name, input, taps, product, rows)?;
        self.place(&format!("{name}.activation"), product, y, rows, activation);
        Ok(())
    }

    fn conv(&mut self, name: &str, x: Signal, y: Signal, window: Window, activation: Activation) -> Result<()> {
        self.columns(&format!("{name}.columns"), x, window);
        self.project(name, x.width, window.taps, y, window.rows, activation)
    }

    fn res2net(&mut self, layer: usize, hidden: Signal, layers: Signal, c: &SpeakerEncoderConfig) -> Result<()> {
        use Activation::{Identity, Relu, Sigmoid};
        let name = format!("blocks.{layer}");
        let projected = Signal::new("projected", hidden.width);
        let res2 = Signal::new("res2", hidden.width);
        let width = hidden.width / c.enc_res2net_scale;
        self.conv(&format!("{name}.tdnn1.conv"), hidden, projected, POINT, Relu)?;
        self.place(
            &format!("{name}.res2net_block.first"),
            projected.group(width, 0),
            res2.group(width, 0),
            Rows::Frames,
            Identity,
        );
        for i in 1..c.enc_res2net_scale {
            let window = Window {
                taps: c.enc_kernel_sizes[layer],
                dilation: c.enc_dilations[layer],
                previous_group: (i >= 2).then_some((i - 1) * width),
                ..POINT
            };
            self.conv(
                &format!("{name}.res2net_block.blocks.{}.conv", i - 1),
                projected.group(width, i),
                res2.group(width, i),
                window,
                Relu,
            )?;
        }
        self.conv(&format!("{name}.tdnn2.conv"), res2, projected, POINT, Relu)?;
        let mean = Signal::new("channel_mean", hidden.width);
        self.g.launch(
            &format!("{name}.se_block.mean"),
            "ecapa_squeeze",
            by_channel(hidden.width),
            THREADS,
            vec![inb(projected.name), outb(mean.name), Rows::Frames.arg(), i32a(hidden.width)],
        );
        let squeezed = Signal::new("squeezed", c.enc_se_channels);
        let gate = Signal::new("gate", hidden.width);
        self.conv(&format!("{name}.se_block.conv1"), mean, squeezed, POOLED, Relu)?;
        self.conv(&format!("{name}.se_block.conv2"), squeezed, gate, POOLED, Sigmoid)?;
        let aggregate = if layers.start == 0 { outb(layers.name) } else { io(layers.name) };
        self.g.launch(
            &format!("{name}.se_block.gate"),
            "ecapa_excite",
            Rows::Frames.grid(hidden.width),
            THREADS,
            vec![
                inb(projected.name),
                inb(gate.name),
                io(hidden.name),
                aggregate,
                Rows::Frames.arg(),
                i32a(hidden.width),
                i32a(layers.stride),
                i32a(layers.start),
            ],
        );
        Ok(())
    }

    fn statistics(&mut self, name: &str, x: Signal, attention: Option<Signal>) {
        self.g.launch(
            name,
            "ecapa_statistics",
            by_channel(x.width),
            THREADS,
            vec![
                inb(x.name),
                inb(attention.unwrap_or(x).name),
                outb("statistics"),
                Rows::Frames.arg(),
                i32a(x.width),
                i32a(i32::from(attention.is_some())),
            ],
        );
    }

    fn pool(&mut self, features: Signal, attention: Signal, speaker: Signal) -> Result<()> {
        use Activation::{Identity, ReluTanh};
        let rows = Rows::Frames;
        let width = features.width;
        self.statistics("asp.global", features, None);
        let columns = packed(3 * width);
        self.columns = self.columns.max(columns);
        self.g.launch(
            "asp.context",
            "ecapa_context",
            rows.grid(columns),
            THREADS,
            vec![inb(features.name), inb("statistics"), outb("columns"), rows.arg(), i32a(width), i32a(columns)],
        );
        self.project("asp.tdnn.conv", 3 * width, 1, attention, rows, ReluTanh)?;
        let probabilities = Signal::new("probabilities", width);
        self.conv("asp.conv", attention, probabilities, POINT, Identity)?;
        self.g.launch(
            "asp.softmax",
            "ecapa_softmax",
            by_channel(width),
            THREADS,
            vec![io(probabilities.name), rows.arg(), i32a(width)],
        );
        self.statistics("asp.weighted", features, Some(probabilities));
        let stats = Signal::new("statistics", 2 * width);
        self.columns("fc.columns", stats, POOLED);
        self.affine("fc", stats.width, 1, speaker, Rows::One)
    }
}

fn by_channel(width: usize) -> [Value; 3] {
    [json!(width.div_ceil(32)), json!(1), json!(1)]
}

fn program(checkpoint: &File, c: &SpeakerEncoderConfig, max_frames: usize, sha: &str) -> Result<(Value, HostTensors)> {
    let mut p = Program { g: Gen::default(), checkpoint, columns: 0, product: 0 };
    let last = c.enc_channels.len() - 1;
    let width = c.enc_channels[0];
    let mfa_width = c.enc_channels[last];
    let mel = Signal::new("mel", c.mel_dim);
    let hidden = Signal::new("hidden", width);
    let aggregate = Signal::new("aggregate", mfa_width);
    let features = Signal::new("features", mfa_width);
    let speaker = Signal::new("speaker", c.enc_dim);
    let window = |i: usize| Window { taps: c.enc_kernel_sizes[i], dilation: c.enc_dilations[i], ..POINT };
    p.conv("blocks.0.conv", mel, hidden, window(0), Activation::Relu)?;
    for i in 1..last {
        p.res2net(i, hidden, aggregate.group(width, i - 1), c)?;
    }
    p.conv("mfa.conv", aggregate, features, window(last), Activation::Relu)?;
    p.pool(features, Signal::new("attention", c.enc_attention_channels), speaker)?;

    ensure!(
        max_frames <= i32::MAX as usize / p.columns.max(p.product),
        "speaker encoder frame cap {max_frames} is too large"
    );
    let mut g = p.g;
    g.buffer(mel.name, "bf16", json!(["frames", mel.width]), "input");
    g.buffer(speaker.name, "bf16", json!([speaker.width]), "output");
    for (name, width) in [
        ("columns", p.columns),
        ("product", p.product),
        ("hidden", width),
        ("projected", width),
        ("res2", width),
        ("aggregate", mfa_width),
        ("features", mfa_width),
        ("attention", c.enc_attention_channels),
        ("probabilities", mfa_width),
    ] {
        g.buffer(name, "bf16", json!(["frames", width]), "workspace");
    }
    for (name, width) in
        [("channel_mean", width), ("gate", width), ("squeezed", c.enc_se_channels), ("statistics", 2 * mfa_width)]
    {
        g.buffer(name, "bf16", json!([width]), "workspace");
    }
    let mut programs = serde_json::Map::from_iter([("speaker_encoder".into(), json!({"calls": g.take()}))]);
    g.finish(&mut programs);
    let (buffers, ops, tensors) = g.into_parts();
    let manifest = json!({
        "schema_version": 5,
        "model": "qwen3-tts-speaker-encoder",
        "vars": {"frames": {"max": max_frames}},
        "states": {},
        "buffers": buffers,
        "modules": {"ecapa": {"source": format!("ecapa-{}.cubin", &sha[..12]), "sha256": sha}},
        "ops": ops,
        "programs": programs,
    });
    Ok((manifest, tensors))
}
