use rmip_demo::harness::{best_of, verifier_query_support};
use rmip_demo::mnist_survey::{load_selection, CertifiedSparseInstance};
use rmip_frontend::bits::BitVec;
use rmip_frontend::circuit::OracleCircuit;
use rmip_frontend::mnist::{SparseMajorityFamily, SparseTableOracle, LABEL_BITS};
use rmip_frontend::{deoracle, layout, loc, trace};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const HELP: &str =
    "mnist_direct_eval — time C^O(x) evaluated directly, per image, for the MNIST circuit at one view count

Usage: cargo run --release --locked --manifest-path tools/mnist-direct-eval/Cargo.toml -- [OPTIONS]
  --selection PATH                 selected.json from tools/mnist/survey_sparse.py (required)
  --expected-selection-sha256 HEX  SHA-256 the bundle must have (required); the tracked
                                   manifest.json beside it records it under
                                   artifacts.\"selected.json\".sha256
  --images N                       First N selected images, in recorded order (default: all)
  --probes K                       Timed batches kept per image; K + 1 run and the first
                                   is discarded (default: 9)
  --output PATH                    JSON written here, atomically, after every image (required)
  --help                           Print this help

One evaluation is OracleCircuit::eval on the accepted label against the frozen judge
table; the batch size is calibrated x4 from 1 until one batch takes at least 2 ms.
Each image's circuit, input and oracle are built as build_case builds them, without
its R1CS or holographic index; case_setup_ms times that build.
";

struct Args {
    selection: PathBuf,
    expected: String,
    images: Option<usize>,
    probes: usize,
    output: PathBuf,
}

fn arguments() -> Result<Option<Args>, String> {
    let (mut selection, mut expected, mut output) = (None, None, None);
    let (mut images, mut probes) = (None, 9usize);
    let mut input = std::env::args().skip(1);
    while let Some(key) = input.next() {
        if key == "--help" || key == "-h" {
            print!("{HELP}");
            return Ok(None);
        }
        let value = input
            .next()
            .ok_or_else(|| format!("missing value for {key}"))?;
        match key.as_str() {
            "--selection" => selection = Some(PathBuf::from(value)),
            "--expected-selection-sha256" => expected = Some(value),
            "--images" => {
                let n: usize = value.parse().map_err(|_| "images must be an integer")?;
                if n == 0 {
                    return Err("images must be at least 1".into());
                }
                images = Some(n);
            }
            "--probes" => {
                probes = value.parse().map_err(|_| "probes must be an integer")?;
                if !(1..=1000).contains(&probes) {
                    return Err("probes must be in 1..=1000".into());
                }
            }
            "--output" => output = Some(PathBuf::from(value)),
            _ => return Err(format!("unknown option {key}")),
        }
    }
    Ok(Some(Args {
        selection: selection.ok_or("--selection is required")?,
        expected: expected.ok_or("--expected-selection-sha256 is required")?,
        images,
        probes,
        output: output.ok_or("--output is required")?,
    }))
}

fn us(d: Duration) -> f64 {
    d.as_nanos() as f64 / 1e3
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1e3
}

fn median(samples: &[Duration]) -> Duration {
    assert!(!samples.is_empty(), "at least one sample");
    let mut sorted = samples.to_vec();
    sorted.sort_unstable();
    let mid = sorted.len() / 2;
    if sorted.len().is_multiple_of(2) {
        (sorted[mid - 1] + sorted[mid]) / 2
    } else {
        sorted[mid]
    }
}

fn spread(values: &[Duration]) -> Value {
    json!({
        "median": us(median(values)),
        "min": us(best_of(values)),
        "max": us(*values.iter().max().expect("at least one sample")),
    })
}

struct DirectCase {
    circuit: OracleCircuit,
    x: BitVec,
    oracle: SparseTableOracle,
    ell: usize,
    distinct_queries: usize,
    trace_nu: usize,
    label: u8,
    threshold: usize,
    masks: Vec<u32>,
}

fn direct_case(instance: &CertifiedSparseInstance) -> Result<DirectCase, String> {
    let cert = &instance.certificate;
    if cert.certified_flips == 0 {
        return Err("the MNIST circuit's MIP requires a positive robustness radius".into());
    }
    let accepted = cert
        .accepted_label
        .ok_or("the MNIST table has no accepting majority label")?;
    let oracle = SparseTableOracle {
        image_id: instance.image_id,
        predicted: instance.predictions().to_vec(),
    };
    let circuit = SparseMajorityFamily::new(instance.image_id, cert.r).build();
    let y = usize::from(accepted);
    let x = BitVec::from_bools((0..LABEL_BITS).map(|k| (y >> k) & 1 == 1).collect());
    let (dc, sites) = deoracle::deoraclize(&circuit);
    let lay = layout::TraceLayout::new(&dc);
    let l = loc::build(&sites, &lay);
    let (accepts, a) = circuit.eval_with_answers(x.as_slice(), &oracle);
    if !accepts {
        return Err(format!(
            "image {}: the yes-instance must accept",
            instance.image_id
        ));
    }
    let z = trace::witness_trace(&dc, &lay, x.as_slice(), a.as_slice());
    let distinct_queries = verifier_query_support(l, &rmip_iop::mle::lift(z.as_slice()));
    Ok(DirectCase {
        circuit,
        x,
        oracle,
        ell: cert.r,
        distinct_queries,
        trace_nu: lay.nu,
        label: accepted,
        threshold: cert.threshold,
        masks: (0..cert.r).map(|m| m as u32).collect(),
    })
}

struct Timing {
    batch: usize,
    kept: Vec<Duration>,
}

fn time_direct(c: &DirectCase, probes: usize) -> Timing {
    time_batched(probes, || c.circuit.eval(c.x.as_slice(), &c.oracle))
}

fn time_tight(c: &DirectCase, probes: usize) -> Timing {
    time_batched(probes, || {
        rmip_frontend::mnist::majority_direct(
            &c.oracle.predicted,
            std::hint::black_box(&c.masks),
            c.label,
            c.threshold,
        )
    })
}

fn time_tight_in_order(c: &DirectCase, probes: usize) -> Timing {
    time_batched(probes, || {
        rmip_frontend::mnist::majority_direct_in_order(
            std::hint::black_box(&c.oracle.predicted),
            c.ell,
            c.label,
            c.threshold,
        )
    })
}

fn time_batched(probes: usize, mut f: impl FnMut() -> bool) -> Timing {
    let mut batch = 1usize;
    loop {
        let t0 = Instant::now();
        for _ in 0..batch {
            std::hint::black_box(f());
        }
        if t0.elapsed() >= Duration::from_millis(2) || batch >= 1 << 20 {
            break;
        }
        batch *= 4;
    }
    let mut cs = Vec::with_capacity(probes + 1);
    for _ in 0..=probes {
        let t0 = Instant::now();
        for _ in 0..batch {
            std::hint::black_box(f());
        }
        cs.push(t0.elapsed() / batch as u32);
    }
    cs.remove(0);
    Timing { batch, kept: cs }
}

fn command_output(command: &str, args: &[&str]) -> Option<String> {
    std::process::Command::new(command)
        .args(args)
        .output()
        .ok()
        .filter(|r| r.status.success())
        .map(|r| String::from_utf8_lossy(&r.stdout).trim().to_owned())
}

fn atomic_json(path: &Path, value: &Value) -> Result<(), String> {
    let mut bytes = serde_json::to_vec_pretty(value).map_err(|e| e.to_string())?;
    bytes.push(b'\n');
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    std::fs::write(&temporary, bytes).map_err(|e| e.to_string())?;
    std::fs::rename(&temporary, path).map_err(|e| e.to_string())
}

fn run(args: Args) -> Result<(), String> {
    let started = Instant::now();
    let data = load_selection(&args.selection, Some(&args.expected))?;
    let replay_ms = ms(started.elapsed());
    let images = args.images.unwrap_or(data.instances.len());
    if images > data.instances.len() {
        return Err(format!(
            "requested {images} images but the selection contains {}",
            data.instances.len()
        ));
    }
    if let Some(directory) = args.output.parent().filter(|d| !d.as_os_str().is_empty()) {
        std::fs::create_dir_all(directory).map_err(|e| e.to_string())?;
    }
    let r = data.manifest.r;
    let mut doc = json!({
        "schema": "mnist-direct-eval-v1", "status": "running",
        "circuit": "original_sparse_mnist_majority_v1",
        "quantity": "C^O(x) evaluated directly: OracleCircuit::eval on x, the accepted label's four bits, against the frozen judge table, one SparseTableOracle lookup per oracle gate through the gate interpreter. Per image the circuit, x and the table oracle are built as CertifiedSparseInstance::build_case builds them, without the rest of its Case (no R1CS, no witness Instance, no holographic index); case_setup_ms is that build's wall time, together with the de-oracled layout, loc and honest trace that trace_nu and distinct_queries are read from",
        "estimator": "batch size calibrated x4 from 1 until one batch takes >= 2 ms (cap 2^20); a probe is one batch, its time divided by the batch size; probes + 1 batches per image, the first discarded; best_us is their minimum (rmip_demo::harness::best_of), median_us their median (mean of the two middle values for an even count)",
        "memoised": "memoised_best_us is best_us when every oracle query is distinct (distinct_queries == ell), where a memoised evaluation (compute - one_call * (ell - support)) would subtract nothing; null otherwise",
        "tight": "tight_best_us / tight_median_us: the same estimator on the direct loop rmip_frontend::mnist::majority_direct (one table lookup per view through the hardwired mask list, then the strict majority), no gate interpreter; tight_in_order_best_us: majority_direct_in_order, the scan the mask list 0..r makes it. Table 1's direct-evaluation column and the paper's multiples use tight_in_order_best_us; tight_best_us and best_us (the interpreter) are kept as checks that the three agree",
        "selection_path": args.selection, "selection_sha256": data.manifest_sha256,
        "views": r, "requested_images": images, "probes": args.probes,
        "certificate_replay_all_selected_ms": replay_ms,
        "reproducibility": {
            "command_argv": std::env::args().collect::<Vec<_>>(),
            "rust_source_fingerprint": rmip_demo::source_fingerprint(),
            "runner": command_output("uname", &["-srm"]),
            "compiler": command_output("rustc", &["--version"]),
            "profile": if cfg!(debug_assertions) { "debug" } else { "release" },
        },
        "images": [], "completed_images": 0, "summary": null,
    });
    atomic_json(&args.output, &doc)?;
    let (mut rows, mut bests, mut medians) = (Vec::new(), Vec::new(), Vec::new());
    let (mut tight_bests, mut in_order_bests) = (Vec::new(), Vec::new());
    for (ordinal, instance) in data.instances.iter().take(images).enumerate() {
        let setup = Instant::now();
        let case = direct_case(instance)?;
        let setup_ms = ms(setup.elapsed());
        let timing = time_direct(&case, args.probes);
        let best = best_of(&timing.kept);
        let middle = median(&timing.kept);
        let tight = time_tight(&case, args.probes);
        let tight_best = best_of(&tight.kept);
        let tight_middle = median(&tight.kept);
        let in_order = time_tight_in_order(&case, args.probes);
        let in_order_best = best_of(&in_order.kept);
        assert_eq!(
            case.circuit.eval(case.x.as_slice(), &case.oracle),
            rmip_frontend::mnist::majority_direct(
                &case.oracle.predicted,
                &case.masks,
                case.label,
                case.threshold
            ),
            "the direct loop agrees with the interpreter"
        );
        eprintln!(
            "R={r}: image {} ({}/{images}): interpreter best {:.1} us, median {:.1} us, batch {}; tight best {:.2} us ({:.2} ns per view), in order {:.2} us",
            instance.image_id,
            ordinal + 1,
            us(best),
            us(middle),
            timing.batch,
            us(tight_best),
            us(tight_best) * 1000.0 / r as f64,
            us(in_order_best)
        );
        rows.push(json!({
            "image_id": instance.image_id, "true_label": instance.true_label,
            "ell": case.ell, "distinct_queries": case.distinct_queries,
            "gates": case.circuit.gates.len(), "trace_nu": case.trace_nu,
            "case_setup_ms": setup_ms, "batch": timing.batch,
            "samples_us": timing.kept.iter().map(|&d| us(d)).collect::<Vec<_>>(),
            "best_us": us(best), "median_us": us(middle),
            "memoised_best_us": (case.distinct_queries == case.ell).then(|| us(best)),
            "tight_batch": tight.batch, "tight_best_us": us(tight_best), "tight_median_us": us(tight_middle),
            "tight_samples_us": tight.kept.iter().map(|&d| us(d)).collect::<Vec<_>>(),
            "tight_in_order_best_us": us(in_order_best),
        }));
        bests.push(best);
        medians.push(middle);
        tight_bests.push(tight_best);
        in_order_bests.push(in_order_best);
        doc["images"] = json!(rows);
        doc["completed_images"] = json!(ordinal + 1);
        doc["summary"] = json!({
            "images": bests.len(), "best_us": spread(&bests), "median_us": spread(&medians),
            "tight_best_us": spread(&tight_bests), "tight_in_order_best_us": spread(&in_order_bests),
        });
        atomic_json(&args.output, &doc)?;
    }
    doc["status"] = json!("completed");
    doc["wall_ms"] = json!(ms(started.elapsed()));
    atomic_json(&args.output, &doc)?;
    println!(
        "{}",
        json!({"status": doc["status"], "views": r, "completed_images": images,
        "summary": doc["summary"], "output": args.output})
    );
    Ok(())
}

fn main() {
    if let Err(error) = arguments().and_then(|args| args.map_or(Ok(()), run)) {
        eprintln!("mnist_direct_eval: {error}");
        std::process::exit(2);
    }
}
