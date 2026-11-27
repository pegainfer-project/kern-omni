use omni_kern::Gen;
use omni_kern::i32a;
use omni_kern::inb;
use omni_kern::ini;
use omni_kern::outb;
use omni_kern::state_io;
use omni_kern::stride;
use serde_json::Value;
use serde_json::json;

/// The builder's output is still a manifest the current kern verifier accepts.
#[test]
fn generated_manifest_verifies() {
    let mut g = Gen::default();
    let w = g.weight("w", &[8], &[0.0; 8]);
    g.buffer("x", "bf16", json!(["rows", 8]), "input");
    g.buffer("y", "bf16", json!(["rows", 8]), "output");
    g.launch("copy", "toy_copy", [json!("rows"), json!(1), json!(1)], 64, vec![inb("x"), inb(&w), outb("y"), i32a(8)]);
    let calls = g.take();

    let mut programs = serde_json::Map::new();
    programs.insert("copy".into(), json!({"batch": {"groups": 1, "rows": 1}, "graph": true, "calls": calls}));
    g.finish(&mut programs);
    let (buffers, ops, _) = g.into_parts();
    let modules: serde_json::Map<String, Value> =
        [("toy".into(), json!({"source": "toy.cubin", "sha256": "00".repeat(32)}))].into_iter().collect();
    let manifest = json!({
        "schema_version": 7,
        "model": "omni-kern-test",
        "vars": {"rows": {"max": 8}},
        "states": {},
        "buffers": buffers,
        "modules": modules,
        "ops": ops,
        "programs": programs,
    });

    let parsed = kern_manifest::Manifest::from_json(&manifest.to_string()).unwrap();
    if let Err(errs) = kern_manifest::verify(parsed) {
        panic!("generated manifest failed verification:\n{}", errs.join("\n"));
    }
}

/// The paged-state and stride constructs every speech manifest uses still
/// verify under the current schema.
#[test]
fn paged_state_manifest_verifies() {
    let mut g = Gen::default();
    let state = g.region(32);
    let w = g.weight("w", &[8], &[0.0; 8]);
    g.buffer("x", "bf16", json!(["rows", 8]), "input");
    g.buffer("y", "bf16", json!(["rows", 8]), "output");
    g.input("t_slot", json!(["rows"]), json!({"index_into": "kv0"}));
    g.input("kv_indptr", json!([3]), json!({"min": 0, "monotone": true}));
    g.input("kv_pages", json!([4]), json!({"index_into": "kv0", "stride": 16}));
    g.launch(
        "copy",
        "toy_copy",
        [json!("rows"), json!(1), json!(1)],
        64,
        vec![
            inb("x"),
            inb(&w),
            outb("y"),
            state_io(state),
            ("inout state", json!({"state": "kv0"})),
            stride(),
            ini("t_slot"),
            ini("kv_indptr"),
            ini("kv_pages"),
        ],
    );
    let calls = g.take();

    let mut programs = serde_json::Map::new();
    programs.insert("copy".into(), json!({"batch": {"groups": 1, "rows": 1}, "graph": true, "calls": calls}));
    let s = g.finish(&mut programs);
    let (buffers, ops, _) = g.into_parts();
    let modules: serde_json::Map<String, Value> =
        [("toy".into(), json!({"source": "toy.cubin", "sha256": "00".repeat(32)}))].into_iter().collect();
    let manifest = json!({
        "schema_version": 7,
        "model": "omni-kern-test",
        "vars": {"rows": {"max": 8}},
        "states": {"kv0": {"bytes_per_token": 256}, "seq": {"bytes_per_seq": s}},
        "buffers": buffers,
        "modules": modules,
        "ops": ops,
        "programs": programs,
    });

    let parsed = kern_manifest::Manifest::from_json(&manifest.to_string()).unwrap();
    if let Err(errs) = kern_manifest::verify(parsed) {
        panic!("paged manifest failed verification:\n{}", errs.join("\n"));
    }
}
