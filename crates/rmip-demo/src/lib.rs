pub mod fixtures;
pub mod harness;
pub mod measured_execution;
pub mod mnist_bench;
pub mod mnist_survey;
pub mod party_metrics;
pub mod stats;
pub mod weakened;
pub mod wire;

pub fn source_fingerprint() -> String {
    use sha2::{Digest, Sha256};
    let Some(repo) = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
    else {
        return "unavailable".into();
    };

    let Ok(out) = std::process::Command::new("git")
        .arg("-C")
        .arg(repo)
        .args(["ls-files", "-z", "--", "crates"])
        .output()
    else {
        return "unavailable".into();
    };
    if !out.status.success() {
        return "unavailable".into();
    }

    let mut paths = std::collections::BTreeSet::new();
    for raw in out
        .stdout
        .split(|&b| b == 0)
        .filter(|s| s.ends_with(b".rs"))
    {
        let Ok(rel) = std::str::from_utf8(raw) else {
            return "unavailable".into();
        };
        paths.insert(rel.to_owned());
    }
    let Ok(declared) = std::fs::read_to_string(repo.join("benchmarks/source-inputs.txt")) else {
        return "unavailable".into();
    };
    for line in declared.lines().map(str::trim) {
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if !line.starts_with("crates/")
            || !line.ends_with(".rs")
            || line
                .split('/')
                .any(|part| part.is_empty() || part == "." || part == "..")
        {
            return "unavailable".into();
        }
        paths.insert(line.to_owned());
    }
    let mut files: Vec<_> = paths
        .into_iter()
        .map(|rel| {
            let key = rel.split('/').map(str::to_owned).collect::<Vec<String>>();
            (key, repo.join(rel))
        })
        .collect();
    if files.is_empty() {
        return "unavailable".into();
    }
    files.sort_by(|a, b| a.0.cmp(&b.0));

    let mut h = Sha256::new();
    for (_, f) in &files {
        let Ok(bytes) = std::fs::read(f) else {
            return "unavailable".into();
        };
        h.update(f.file_name().unwrap_or_default().as_encoded_bytes());
        h.update(bytes);
    }
    format!("{:x}", h.finalize())[..16].to_string()
}
