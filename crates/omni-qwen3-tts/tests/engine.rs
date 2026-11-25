use std::collections::BTreeMap;
use std::path::Path;
use std::path::PathBuf;
use std::sync::atomic::Ordering;
use std::time::Duration;

use bytes::Bytes;
use omni_engine::Draft;
use omni_engine::Event;
use omni_engine::Finish;
use omni_engine::Handle;
use omni_engine::ReferencePcm;
use omni_engine::Rejected;
use omni_engine::Speech;
use omni_qwen3_tts::config::Config;
use omni_qwen3_tts::engine;
use omni_qwen3_tts::engine::Options;
use omni_qwen3_tts::engine::chunk_due;
use omni_qwen3_tts::engine::frame_cap;
use omni_qwen3_tts::weights::File;
use proptest::prelude::*;
use tokio::sync::mpsc::UnboundedReceiver;

/// Chunks a request emits as its frames arrive one per step, then as it finishes.
fn chunks(total: usize, first: usize, steady: usize) -> Vec<usize> {
    let mut out = Vec::new();
    let mut emitted = 0;
    for generated in 0..=total {
        while let Some(n) = chunk_due(generated, emitted, generated == total, first, steady) {
            out.push(n);
            emitted += n;
        }
    }
    out
}

proptest! {
    #[test]
    fn every_frame_is_emitted_once_in_bounded_chunks(total in 0usize..200, first in 1usize..16, steady in 1usize..32) {
        let c = chunks(total, first, steady);
        prop_assert_eq!(c.iter().sum::<usize>(), total);
        prop_assert!(c.first().is_none_or(|&n| n <= first));
        prop_assert!(c.iter().skip(1).all(|&n| n <= steady));
        prop_assert!(c.iter().all(|&n| n > 0));
    }

    #[test]
    fn only_the_last_chunk_is_short(total in 1usize..200, first in 1usize..16, steady in 1usize..32) {
        let c = chunks(total, first, steady);
        let full = |i: usize, n: usize| n == if i == 0 { first } else { steady };
        prop_assert!(c[..c.len() - 1].iter().enumerate().all(|(i, &n)| full(i, n)));
    }

    #[test]
    fn frame_cap_respects_the_model(chars in 0usize..10_000, max in 1usize..10_000) {
        prop_assert!(frame_cap(chars, max) <= max);
    }
}

/// The released fixtures keep the original prepared WAVs beside `fixtures/`.
/// These original PCM bytes enter the channel; their signed-PCM normalization
/// must reproduce the exact samples used by the official speaker encoder.
fn reference(fixtures: &Path, case: &str) -> Bytes {
    let path = fixtures.parent().unwrap().join("prepared").join(format!("{case}.wav"));
    let wav = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    assert_eq!(&wav[..4], b"RIFF");
    assert_eq!(&wav[8..16], b"WAVEfmt ");
    let u16_at = |i| u16::from_le_bytes(wav[i..i + 2].try_into().unwrap());
    let u32_at = |i| u32::from_le_bytes(wav[i..i + 4].try_into().unwrap());
    assert_eq!(u32_at(16), 16, "prepared WAV has a PCM format chunk");
    assert_eq!((u16_at(20), u16_at(22), u32_at(24), u16_at(34)), (1, 1, 24_000, 16));
    assert_eq!(&wav[36..40], b"data");
    let pcm = &wav[44..];
    assert_eq!(pcm.len(), u32_at(40) as usize);
    let normalized: Vec<_> =
        pcm.as_chunks::<2>().0.iter().map(|&x| f32::from(i16::from_le_bytes(x)) / 32768.0).collect();
    let golden = File::open(&fixtures.join(format!("{case}-xvector-true.safetensors"))).unwrap();
    assert_eq!(normalized, golden.host("ref_pcm").unwrap().data, "{case}: PCM16 normalization");
    Bytes::copy_from_slice(pcm)
}

fn clone_speech(handle: &Handle, pcm: &Bytes, input: &str) -> Speech {
    handle
        .info
        .check(Draft {
            input: input.into(),
            voice: "alloy".into(),
            extra: BTreeMap::from([("language".into(), "english".into()), ("seed".into(), 0.into())]),
            ref_audio: Some(ReferencePcm { pcm: pcm.clone(), sample_rate: 24_000 }),
            x_vector_only_mode: Some(true),
            ..Draft::default()
        })
        .unwrap()
}

async fn next(events: &mut UnboundedReceiver<Event>) -> Event {
    tokio::time::timeout(Duration::from_secs(30), events.recv())
        .await
        .expect("the engine must keep making progress")
        .expect("the engine must send Done before closing the stream")
}

fn audio(event: Event) -> Bytes {
    let Event::Audio(pcm) = event else { panic!("expected audio, got {event:?}") };
    assert!(!pcm.is_empty() && pcm.len().is_multiple_of(2), "whole nonempty PCM16 samples");
    pcm
}

async fn complete(events: &mut UnboundedReceiver<Event>, first: Bytes, samples_per_frame: usize) {
    let mut pcm = first.to_vec();
    loop {
        match next(events).await {
            event @ Event::Audio(_) => pcm.extend_from_slice(&audio(event)),
            Event::Done(done) => {
                assert_eq!(done.finish, Finish::Complete);
                assert!(done.input_units > 0 && done.frames > 0);
                assert_eq!(pcm.len(), done.frames as usize * samples_per_frame * 2);
                assert!(pcm.iter().any(|&x| x != 0), "synthesis must produce a nonzero waveform");
                let end = tokio::time::timeout(Duration::from_secs(30), events.recv())
                    .await
                    .expect("Done must close the stream");
                assert!(end.is_none(), "exactly one Done terminates a stream");
                return;
            }
        }
    }
}

async fn occupancy(handle: &Handle, ready: impl Fn(usize, usize) -> bool) {
    tokio::time::timeout(Duration::from_secs(30), async {
        loop {
            let waiting = handle.load.waiting.load(Ordering::Relaxed);
            let running = handle.load.running.load(Ordering::Relaxed);
            assert!(waiting <= 2, "reference preparation and ready backlog must stay bounded: {waiting}");
            assert!(running <= 2, "batch capacity: {running}");
            if ready(waiting, running) {
                return;
            }
            tokio::time::sleep(Duration::from_millis(1)).await;
        }
    })
    .await
    .expect("the engine must release canceled work and admit later requests");
}

/// Exercises the channel used by the HTTP service, including its loading
/// thread. Opt in with the same Base checkpoint and fixture bundle as golden.
#[tokio::test]
async fn base_serves_mixed_references_and_releases_canceled_requests() {
    let get = |key| std::env::var_os(key).map(PathBuf::from);
    let Some((dir, fixtures)) = get("OMNI_QWEN3_TTS_BASE_MODEL").zip(get("OMNI_QWEN3_TTS_BASE_FIXTURES")) else {
        eprintln!("skipped: set OMNI_QWEN3_TTS_BASE_MODEL and OMNI_QWEN3_TTS_BASE_FIXTURES");
        return;
    };
    let english = reference(&fixtures, "en-short");
    let chinese = reference(&fixtures, "cn-short");
    let longest: Bytes = english.iter().copied().cycle().take(30 * 24_000 * 2).collect();
    let samples_per_frame = Config::load(&dir).unwrap().samples_per_frame;
    let queue = 2;
    let opts = Options { max_batch: 2, max_step_tokens: 512, kv_gib: 0.25, first_chunk_frames: 1, chunk_frames: 1 };
    let (handle, thread) = engine::start(0, dir, opts, "base-test".into(), 2048, queue).unwrap();
    let info = &handle.info;
    assert!(info.voices.is_empty());
    let capability = info.reference_audio.as_ref().expect("Base advertises reference audio");
    assert_eq!(capability.sample_rate, 24_000);
    assert_eq!(capability.duration_seconds, 1..=30);
    assert!(capability.audio_only && !capability.with_transcript);
    let missing = info.check(Draft { input: "Hello.".into(), voice: "alloy".into(), ..Draft::default() });
    assert_eq!(missing.unwrap_err().param, "ref_audio");
    let icl = info.check(Draft {
        input: "Hello.".into(),
        voice: "alloy".into(),
        ref_audio: Some(ReferencePcm { pcm: english.clone(), sample_rate: 24_000 }),
        ref_text: Some("The reference transcript.".into()),
        x_vector_only_mode: Some(false),
        ..Draft::default()
    });
    assert_eq!(icl.unwrap_err().param, "x_vector_only_mode");

    let input = "Keep speaking while another reference is prepared. Both requests use the same words and seed \
                 with different voices.";
    let mut a = handle.submit(clone_speech(&handle, &english, input)).unwrap();
    let mut b = handle.submit(clone_speech(&handle, &chinese, input)).unwrap();
    let first_a = audio(next(&mut a).await);
    let first_b = audio(next(&mut b).await);
    assert_ne!(first_a, first_b, "each row must use its own reference speaker");
    occupancy(&handle, |_, running| running == 2).await;

    // With both sequence slots occupied, only two references may leave the
    // channel. Cancel the full backlog, including the longest supported PCM.
    let queued = clone_speech(&handle, &longest, "Hello.");
    let mut pending = Vec::new();
    loop {
        match handle.submit(queued.clone()) {
            Ok(events) => pending.push(events),
            Err(Rejected::Full) => break,
            Err(e) => panic!("the live engine refused a checked request: {e}"),
        }
        assert!(pending.len() <= 2 + queue, "the internal reference backlog must be bounded");
    }
    assert!(!pending.is_empty());
    occupancy(&handle, |waiting, _| waiting > 0).await;
    drop(pending);
    occupancy(&handle, |waiting, _| waiting == 0).await;

    // Discard already queued chunks before the late request, so the next one
    // proves existing synthesis advances after the long reference arrives.
    while let Ok(event) = b.try_recv() {
        audio(event);
    }
    let mut late = handle.submit(queued).unwrap();
    occupancy(&handle, |waiting, _| waiting > 0).await;
    audio(next(&mut b).await);
    drop(a);
    let first_late = audio(next(&mut late).await);
    drop(b);
    complete(&mut late, first_late, samples_per_frame).await;
    occupancy(&handle, |waiting, running| waiting == 0 && running == 0).await;

    // Cancel the only running request without waking the engine with another
    // submission: its published occupancy must reach idle on its own.
    let mut alone = handle.submit(clone_speech(&handle, &english, input)).unwrap();
    audio(next(&mut alone).await);
    drop(alone);
    occupancy(&handle, |waiting, running| waiting == 0 && running == 0).await;

    // The same transition must release a lone reference before admission.
    let preparing = handle.submit(clone_speech(&handle, &longest, "Hello.")).unwrap();
    occupancy(&handle, |waiting, running| waiting > 0 && running == 0).await;
    drop(preparing);
    occupancy(&handle, |waiting, running| waiting == 0 && running == 0).await;

    let mut after = handle.submit(clone_speech(&handle, &chinese, "Hello.")).unwrap();
    let first_after = audio(next(&mut after).await);
    complete(&mut after, first_after, samples_per_frame).await;
    occupancy(&handle, |waiting, running| waiting == 0 && running == 0).await;
    drop(handle);
    thread.join().expect("the serving thread shuts down after its handle is dropped");
}
