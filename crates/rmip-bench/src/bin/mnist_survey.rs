use rmip_demo::measured_execution::{
    parameters_compiled, run_protocol_opts, Compilation, IopBody, Lookup, MipBackend, RunOptions,
    Soundness,
};
use rmip_demo::mnist_survey::load_selection;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::collections::BTreeSet;
use std::path::{Path, PathBuf};
use std::time::Instant;

#[global_allocator]
static ALLOCATOR: rmip_demo::party_metrics::TrackingAllocator =
    rmip_demo::party_metrics::TrackingAllocator;

const HELP: &str = "mnist_survey — the MNIST circuit on the fixed 50 test images at one view count

Usage: cargo run --release -p rmip-bench --bin mnist_survey -- [OPTIONS]
  --selection PATH       selected.json from tools/mnist/survey_sparse.py
                         Default: benchmarks/mnist-test/artifacts/R256/selected.json
  --expected-selection-sha256 HEX  Pin the trusted selection artifact
  --images N             Number of selected images to run, in recorded order
                         Default: 50; skip + images must not exceed the selection
  --skip N               Skip the first N selected images (default: none), so
                         disjoint shards can run in parallel into ONE output
                         directory. When given, the ledger is written as
                         ledger-skip-NNNNN.json / .csv (N zero-padded to five
                         digits) and holds only the shard's images; per-image
                         files keep their image-NNNNN.json names.
                         summarize_sweep.py merges shard ledgers by image_id.
                         Example: --skip 7 --images 1 runs the eighth image.
  --mode MODE            replay | protocol (default: replay)
  --backend NAME         ladder | flat (default: ladder): the commitment backend
  --body NAME            flat | packed64 | packed128 (default: flat): the relation body
                         measured in protocol mode and parameterised in both
  --protocol-smoke        One actual base execution per image, without amplification
  --record-lookups       Keep every oracle lookup's query and elapsed time in the
                         ledger (a diagnostic that costs the prover about
                         50 ns per view inside its timer; default: count only)
  --single-shot          The single-execution parameters (a 2^-21 sampling budget,
                         a 23-bit opening of 34 columns, pins for a 2^-30 collision
                         term, R = 1) instead of the registry's amplified ones;
                         ladder backend only
  --soundness MODE       amplified | single-shot | single-shot-N: the single-shot
                         mode at a 2^-N per-execution target (single-shot-40 is the
                         paper's 40-bit target)
  --depth D              The ladder's depth (default: the full depth for the trace;
                         0 is the one-table opening, the configuration whose proof
                         is complete); required with a
                         single-shot mode, since the full depth can be 2 or more,
                         where the bound is conditional on the open ladder lemma
  --compilation MODE     two-prover | rewind (default: two-prover): two provers with
                         the checker audited, or one prover that the verifier
                         restores to its posting state before each query (the
                         paper's IOP-to-RIP compilation); rewind runs the ladder at
                         depth 0 without pins and needs --backend ladder --depth 0
  --output-dir PATH      Default: benchmarks/mnist-test/results/latest
  --help                 Print this help

The loader independently replays every exact certificate from the frozen tables.
Protocol mode runs full configured amplification for each image unless explicitly
marked smoke. The ladder at depth 0 reads one bundled tuple of columns (q_long = 1,
padding refused); under --compilation rewind the one prover is run and timed, under
two-prover a checker is audited besides. The original circuit, sparse-query codec and
soundness budgets are unchanged; the coins come from the OS challenger. CNN judging and mask
preparation are offline; online oracle calls are frozen table lookups.
";

struct Args {
    selection: PathBuf,
    expected: Option<String>,
    images: usize,
    skip: Option<usize>,
    mode: String,
    backend: MipBackend,
    body: IopBody,
    smoke: bool,
    soundness: Soundness,
    ladder_depth: Option<usize>,
    compilation: Compilation,
    record_lookups: bool,
    output: PathBuf,
}

fn arguments() -> Result<Option<Args>, String> {
    let mut out = Args {
        selection: "benchmarks/mnist-test/artifacts/R256/selected.json".into(),
        expected: None,
        images: 50,
        skip: None,
        mode: "replay".into(),
        backend: MipBackend::Ladder,
        body: IopBody::Flat,
        smoke: false,
        soundness: Soundness::Amplified,
        ladder_depth: None,
        compilation: Compilation::TwoProver,
        record_lookups: false,
        output: "benchmarks/mnist-test/results/latest".into(),
    };
    let mut input = std::env::args().skip(1);
    while let Some(key) = input.next() {
        if key == "--help" || key == "-h" {
            print!("{HELP}");
            return Ok(None);
        }
        if key == "--protocol-smoke" {
            out.smoke = true;
            continue;
        }
        if key == "--record-lookups" {
            out.record_lookups = true;
            continue;
        }
        if key == "--single-shot" {
            out.soundness = Soundness::SingleShot;
            continue;
        }
        let value = input
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--selection" => out.selection = value.into(),
            "--expected-selection-sha256" => out.expected = Some(value),
            "--images" => {
                out.images = value.parse().map_err(|_| "images must be an integer")?;
                if !(1..=10_000).contains(&out.images) {
                    return Err("images must be in 1..=10000".into());
                }
            }
            "--skip" => {
                let skip: usize = value.parse().map_err(|_| "skip must be an integer")?;
                if skip >= 10_000 {
                    return Err("skip must be in 0..10000".into());
                }
                out.skip = Some(skip);
            }
            "--mode" if ["replay", "protocol"].contains(&value.as_str()) => out.mode = value,
            "--mode" => return Err("mode must be replay or protocol".into()),
            "--backend" => {
                out.backend = MipBackend::parse(&value)
                    .ok_or_else(|| "backend must be ladder or flat".to_string())?
            }
            "--body" => {
                out.body = IopBody::parse(&value)
                    .ok_or_else(|| "body must be flat, packed64 or packed128".to_string())?
            }
            "--soundness" => {
                out.soundness = Soundness::parse(&value).ok_or_else(|| {
                    "soundness must be amplified, single-shot or single-shot-N".to_string()
                })?
            }
            "--depth" => {
                out.ladder_depth = Some(value.parse().map_err(|_| "depth must be an integer")?)
            }
            "--compilation" => {
                out.compilation = Compilation::parse(&value)
                    .ok_or_else(|| "compilation must be two-prover or rewind".to_string())?
            }
            "--output-dir" => out.output = value.into(),
            _ => return Err(format!("unknown option {key}")),
        }
    }
    if out.smoke && out.mode != "protocol" {
        return Err("--protocol-smoke requires --mode protocol".into());
    }
    if out.soundness.is_single_shot()
        && out.backend == MipBackend::Ladder
        && out.ladder_depth.is_none()
    {
        return Err("single-shot modes require an explicit --depth D: 0 and 1 are proved, deeper depths are conditional on the open ladder lemma and the ledger says so".into());
    }
    if out.compilation == Compilation::Rewind
        && (out.backend != MipBackend::Ladder || out.ladder_depth != Some(0))
    {
        return Err(
            "--compilation rewind runs the one-table ladder without pins: pass --backend ladder --depth 0"
                .into(),
        );
    }
    Ok(Some(out))
}

fn command_output(command: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(command)
        .args(args)
        .output()
        .ok()
        .filter(|r| r.status.success())
        .map(|r| String::from_utf8_lossy(&r.stdout).trim().to_owned())
}

fn content_fingerprint() -> Result<String, String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .unwrap()
        .parent()
        .unwrap();
    fn collect(path: &Path, files: &mut Vec<PathBuf>) -> std::io::Result<()> {
        if path.is_dir() {
            for entry in std::fs::read_dir(path)? {
                let entry = entry?;
                if entry.file_name() != "__pycache__" {
                    collect(&entry.path(), files)?;
                }
            }
        } else if path
            .extension()
            .is_some_and(|e| ["rs", "py", "toml", "lock"].iter().any(|s| e == *s))
        {
            files.push(path.to_owned());
        }
        Ok(())
    }
    let mut files = Vec::new();
    collect(&root.join("crates"), &mut files).map_err(|e| e.to_string())?;
    collect(&root.join("tools/mnist"), &mut files).map_err(|e| e.to_string())?;
    files.extend([root.join("Cargo.toml"), root.join("Cargo.lock")]);
    files.sort();
    let mut hash = Sha256::new();
    hash.update(b"mnist-survey-source-v1\0");
    for file in files {
        let relative = file.strip_prefix(root).unwrap().to_string_lossy();
        hash.update((relative.len() as u64).to_le_bytes());
        hash.update(relative.as_bytes());
        let bytes = std::fs::read(file).map_err(|e| e.to_string())?;
        hash.update((bytes.len() as u64).to_le_bytes());
        hash.update(bytes);
    }
    Ok(format!("{:x}", hash.finalize()))
}

fn oracle_measurements(calls: &[Lookup], logical_calls: usize, recorded: bool) -> Value {
    if !recorded {
        return json!({
            "mode": "frozen_sparse_judge_table",
            "logical_calls": logical_calls, "cache_misses": logical_calls,
            "distinct_queries": Value::Null, "serialized_query_bytes": Value::Null, "table_lookup_ms": Value::Null,
            "diagnostics": "lookups counted, not recorded (--record-lookups records them)",
            "query_codec": "original sparse tag 0x02; 8 bytes, 64 bits",
            "cnn_inference": "offline preparation; excluded from online table-lookup latency",
            "resolved_input": "fixed 28x28 image and registered six-nonzero-pixel mask; judge consumes the original two sparse input planes",
        });
    }
    assert_eq!(
        calls.len(),
        logical_calls,
        "the recorded lookups are the counted calls"
    );
    let unique: BTreeSet<&[u8]> = calls.iter().map(|x| x.query.as_slice()).collect();
    json!({
        "mode": "frozen_sparse_judge_table",
        "logical_calls": calls.len(), "cache_misses": calls.len(),
        "distinct_queries": unique.len(),
        "serialized_query_bytes": calls.iter().map(|x| x.query.len()).sum::<usize>(),
        "table_lookup_ms": calls.iter().map(|x| x.elapsed.as_secs_f64()*1000.0).sum::<f64>(),
        "query_codec": "original sparse tag 0x02; 8 bytes, 64 bits",
        "cnn_inference": "offline preparation; excluded from online table-lookup latency",
        "resolved_input": "fixed 28x28 image and registered six-nonzero-pixel mask; judge consumes the original two sparse input planes",
    })
}

fn atomic_json(path: &Path, value: &Value) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    let temporary = path.with_extension("json.tmp");
    std::fs::write(&temporary, bytes).map_err(|e| e.to_string())?;
    std::fs::rename(temporary, path).map_err(|e| e.to_string())
}

fn ledger_stem(skip: Option<usize>) -> String {
    skip.map_or_else(|| "ledger".into(), |s| format!("ledger-skip-{s:05}"))
}

fn write_ledger(directory: &Path, stem: &str, ledger: &Value) -> Result<(), String> {
    atomic_json(&directory.join(format!("{stem}.json")), ledger)?;
    let mut csv = String::from("image_id,true_label,accepted_label,certified_flips,views,threshold,base_t,amplification_repetitions,actual_Q_O,status,prover_ms,verifier_ms,checker_ms,prover_peak_heap_bytes,serialized_communication_bytes,case_setup_ms,backend,q_long,eps_col\n");
    for row in ledger["images"].as_array().unwrap() {
        let c = &row["certificate"];
        let p = &row["protocol_parameters"];
        let e = &row["execution"];
        let values = [
            row["image_id"].clone(),
            row["true_label"].clone(),
            c["accepted_label"].clone(),
            c["certified_flips"].clone(),
            c["r"].clone(),
            c["threshold"].clone(),
            p["base_t"].clone(),
            p["amplification_repetitions"].clone(),
            e["actual_total_Q_O"].clone(),
            row["status"].clone(),
            e["prover_runtime_ms"].clone(),
            e["verifier_runtime_ms"].clone(),
            e["party_measurements"]["checker_prover_including_preparation"]["runtime_ms"].clone(),
            e["prover_peak_memory_bytes"].clone(),
            e["total_serialized_communication_bytes"].clone(),
            row["case_setup_ms"].clone(),
            row["backend"].clone(),
            p["q_long"].clone(),
            row["eps_col"].clone(),
        ];
        csv.push_str(
            &values
                .iter()
                .map(|v| match v {
                    Value::Null => String::new(),
                    Value::String(s) => format!("\"{}\"", s.replace('"', "\"\"")),
                    _ => v.to_string(),
                })
                .collect::<Vec<_>>()
                .join(","),
        );
        csv.push('\n');
    }
    let path = directory.join(format!("{stem}.csv"));
    let temporary = directory.join(format!("{stem}.csv.tmp"));
    std::fs::write(&temporary, csv).map_err(|e| e.to_string())?;
    std::fs::rename(&temporary, &path).map_err(|e| e.to_string())
}

fn run(args: Args) -> Result<(), String> {
    let started = Instant::now();
    let data = load_selection(&args.selection, args.expected.as_deref())?;
    let replay_ms = started.elapsed().as_secs_f64() * 1000.0;
    let skip = args.skip.unwrap_or(0);
    if skip + args.images > data.instances.len() {
        return Err(format!(
            "requested {} images after skipping {} but selection contains {}",
            args.images,
            skip,
            data.instances.len()
        ));
    }
    let stem = ledger_stem(args.skip);
    std::fs::create_dir_all(&args.output).map_err(|e| e.to_string())?;
    let criterion = serde_json::to_value(&data.manifest.criterion).map_err(|e| e.to_string())?;
    let sources = serde_json::to_value(&data.manifest.sources).map_err(|e| e.to_string())?;
    let mut ledger = json!({
        "schema": "mnist-sparse-mip-ledger-v2", "status": "running",
        "backend": args.backend.name(), "soundness": args.soundness.name(), "body": args.body.name(), "skip": skip,
        "compilation": args.compilation.name(),
        "circuit": "sparse_mnist_majority_v2_regular_popcount", "mode": args.mode,
        "selection_path": args.selection, "selection_sha256": data.manifest_sha256,
        "selection_criterion": criterion, "sources": sources,
        "requested_images": args.images, "views": data.manifest.r, "threshold": data.manifest.threshold,
        "certificate_replay_all_selected_ms": replay_ms,
        "preprocessing": "download, CNN inference, mask registry and exact certification occur offline; each per-image circuit/witness/index setup is timed separately. case_setup_ms is the whole per-image case construction (circuit, R1CS, witness trace) and, where the holographic index is built (nu >= 15), includes BOTH offline index halves; index_ms is the offline VERIFIER preprocessing alone (rmip_holo::preprocess_verifier: packed table and its verifier-owned encoding, no prover work); index_prover_ms is the prover's offline index tables alone (rmip_holo::preprocess_prover); both are null where no index is built",
        "robustness": "exact whole-circuit tolerance to corrupted Boolean oracle-answer occurrences for a fixed image, judge, and mask registry",
        "reproducibility": {
            "command_argv": std::env::args().collect::<Vec<_>>(),
            "started_at_unix_seconds": std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).ok().map(|d| d.as_secs()),
            "source_fingerprint": content_fingerprint()?, "rust_source_fingerprint": rmip_demo::source_fingerprint(),
            "source_fingerprint_convention": "SHA256 of on-disk crates Rust/Cargo, tools/mnist Python, and root Cargo files; paths and lengths included; independent of staging",
            "compiler": command_output("rustc", &["--version", "--verbose"]),
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
            "hardware": command_output("uname", &["-srm"]),
            "cpu_model": command_output("sysctl", &["-n", "machdep.cpu.brand_string"]),
            "warmups": 0, "measurements_per_image": 1,
            "execution_threading": "shipped backend defaults; corpus CPU inference is a separate offline process",
            "smoke_only": args.smoke,
        },
        "images": [], "completed_images": 0, "accepted_images": 0,
        "total_base_executions": 0, "actual_total_Q_O": 0,
    });
    write_ledger(&args.output, &stem, &ledger)?;
    let mut rows = Vec::new();
    let mut accepted = 0usize;
    let mut base_runs = 0u64;
    let mut oracle_calls = 0u64;
    for (ordinal, instance) in data
        .instances
        .iter()
        .skip(skip)
        .take(args.images)
        .enumerate()
    {
        eprintln!(
            "MNIST circuit: image {} ({}/{}, after skipping {})",
            instance.image_id,
            ordinal + 1,
            args.images,
            skip
        );
        let setup = Instant::now();
        let (case, oracle) = instance.build_case("sparse-mnist-test")?;
        let setup_ms = setup.elapsed().as_secs_f64() * 1000.0;
        let mut row = json!({
            "image_id": instance.image_id, "true_label": instance.true_label,
            "certificate": &instance.certificate, "case_setup_ms": setup_ms,
            "index_ms": case.t_preprocess_verifier.map(|d| d.as_secs_f64()*1000.0),
            "index_prover_ms": case.t_preprocess_prover.map(|d| d.as_secs_f64()*1000.0),
            "shape": {"ell": case.inst.ell, "b_max": case.inst.b_max(),
                "B_tr": case.inst.b_tr(), "gates": case.circuit.gates.len(),
                "r1cs_rows": case.sys.m, "r1cs_nonzeros": case.sys.z,
                "trace_N": case.layout.n_trace, "trace_nu": case.layout.nu},
            "status": "certificate_replayed_and_case_built",
        });
        let protocol_parameters = parameters_compiled(
            &case,
            args.backend,
            args.soundness,
            args.body,
            args.ladder_depth,
            args.compilation,
        );
        row["backend"] = protocol_parameters["backend"].clone();
        row["eps_col"] = protocol_parameters["eps_col"].clone();
        row["protocol_parameters"] = protocol_parameters;
        if args.mode == "protocol" {
            let measured = run_protocol_opts(
                &case,
                &oracle,
                RunOptions {
                    amplified: !args.smoke,
                    record_lookups: args.record_lookups,
                    soundness: args.soundness,
                    body: args.body,
                    ladder_depth: args.ladder_depth,
                    compilation: args.compilation,
                },
                args.backend,
            )?;
            let ok = measured.measurements["status"] == "accepted";
            accepted += usize::from(ok);
            base_runs += measured.measurements["executed_repetitions"]
                .as_u64()
                .unwrap();
            let verifier_calls =
                measured.measurements["actual_total_Q_O"].as_u64().unwrap() as usize;
            let main_calls = case.inst.ell
                * measured.measurements["executed_repetitions"]
                    .as_u64()
                    .unwrap() as usize;
            oracle_calls += verifier_calls as u64;
            row["execution"] = measured.measurements;
            row["verifier_oracle"] = oracle_measurements(
                &measured.verifier_calls,
                verifier_calls,
                args.record_lookups,
            );
            row["main_prover_oracle"] =
                oracle_measurements(&measured.main_prover_calls, main_calls, args.record_lookups);
            row["status"] = json!(if ok { "accepted" } else { "rejected" });
        }
        let raw_path = args
            .output
            .join(format!("image-{:05}.json", instance.image_id));
        atomic_json(&raw_path, &row)?;
        let raw = std::fs::read(&raw_path).map_err(|e| e.to_string())?;
        row["raw_record_path"] = json!(raw_path);
        row["raw_record_sha256"] = json!(format!("{:x}", Sha256::digest(raw)));
        if let Some(object) = row.get_mut("execution").and_then(Value::as_object_mut) {
            object.remove("raw_base_executions");
        }
        rows.push(row);
        ledger["images"] = json!(rows);
        ledger["completed_images"] = json!(ordinal + 1);
        ledger["accepted_images"] = json!(accepted);
        ledger["total_base_executions"] = json!(base_runs);
        ledger["actual_total_Q_O"] = json!(oracle_calls);
        write_ledger(&args.output, &stem, &ledger)?;
    }
    let ok = args.mode == "replay" || accepted == args.images;
    ledger["status"] = json!(if ok { "completed" } else { "failed" });
    ledger["wall_ms"] = json!(started.elapsed().as_secs_f64() * 1000.0);
    write_ledger(&args.output, &stem, &ledger)?;
    println!(
        "{}",
        json!({"status": ledger["status"], "completed_images": args.images,
        "accepted_images": accepted, "total_base_executions": base_runs,
        "actual_total_Q_O": oracle_calls, "output_dir": args.output,
        "ledger": format!("{stem}.json")})
    );
    if ok {
        Ok(())
    } else {
        Err("one or more honest MIP executions rejected; see ledger".into())
    }
}

fn main() {
    if let Err(error) = arguments().and_then(|args| args.map_or(Ok(()), run)) {
        eprintln!("mnist_survey: {error}");
        std::process::exit(2);
    }
}
