use crate::harness::Case;
use rmip_frontend::bits::BitVec;
use rmip_frontend::mnist::{SparseMajorityFamily, SparseTableOracle, LABEL_BITS};
use rmip_frontend::Ratio;
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::collections::HashSet;
use std::io::Read;
use std::path::Path;

pub const DATASET_COUNT: usize = 10_000;
pub const TEST_IMAGES_SHA256: &str =
    "0fa7898d509279e482958e8ce81c8e77db3f2f8254e26661ceb7762c4d494ce7";
pub const TEST_LABELS_SHA256: &str =
    "ff7bcfd416de33731a308c3f266cc351222c34898ecbeaf847f06e48f7ec33f2";
pub const CHECKPOINT_SHA256: &str =
    "c3b2b2bce5cc572f782118798d89d0fddca6cef0c0e37029e82b4b87b4d330d9";
pub const JUDGE_MODEL_HASH: &str =
    "4f8eb534290fdbc905f20f3afbf19663289df8e369f04785da5a71a92a2c9a41";
pub const MAX_SELECTION_BYTES: u64 = 1024 * 1024 * 1024 * 1024;
pub const SUPPORTED_VIEW_COUNTS: [usize; 17] = [
    256,
    1024,
    4096,
    16384,
    65536,
    262144,
    1048576,
    1 << 21,
    1 << 22,
    1 << 23,
    1 << 24,
    1 << 25,
    1 << 26,
    1 << 27,
    1 << 28,
    1 << 29,
    1 << 30,
];

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn hash_is_valid(hash: &str) -> bool {
    hash.len() == 64
        && hash
            .bytes()
            .all(|b| b.is_ascii_digit() || (b'a'..=b'f').contains(&b))
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize)]
pub struct SparseCertificate {
    pub r: usize,
    pub threshold: usize,
    pub vote_counts: [usize; 10],
    pub flip_distances: [usize; 10],
    pub min_flip_distance: usize,
    pub certified_flips: usize,
    pub accepted_label: Option<u8>,
    pub correctly_accepted: bool,
    pub positive_whole_circuit: bool,
    pub strong_whole_circuit: bool,
    pub strong_correct: bool,
}

pub struct CertifiedSparseInstance {
    data: CertifiedSparseData,
}

pub struct CertifiedSparseData {
    pub image_id: u16,
    pub true_label: u8,
    pub certificate: SparseCertificate,
    predictions: Vec<u8>,
}

impl std::ops::Deref for CertifiedSparseInstance {
    type Target = CertifiedSparseData;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl CertifiedSparseInstance {
    pub fn predictions(&self) -> &[u8] {
        &self.data.predictions
    }

    pub fn build_case(&self, family: &'static str) -> Result<(Case, SparseTableOracle), String> {
        let cert = &self.certificate;
        if cert.certified_flips == 0 {
            return Err("the MNIST circuit's MIP requires a positive robustness radius".into());
        }
        let accepted = cert
            .accepted_label
            .ok_or("the MNIST table has no accepting majority label")?;
        let no_label = (0..10usize)
            .filter(|&y| y != usize::from(accepted))
            .max_by_key(|&y| (cert.vote_counts[y], usize::MAX - y))
            .expect("nine rejected labels");
        let label_bits =
            |y: usize| BitVec::from_bools((0..LABEL_BITS).map(|k| (y >> k) & 1 == 1).collect());
        let oracle = SparseTableOracle {
            image_id: self.image_id,
            predicted: self.predictions().to_vec(),
        };
        let (circuit, shape) = SparseMajorityFamily::new(self.image_id, cert.r).build_with_shape();
        let case = crate::mnist_bench::assemble(
            family,
            circuit,
            Some(shape),
            label_bits(usize::from(accepted)),
            label_bits(no_label),
            Ratio::new(cert.certified_flips as u64, cert.r as u64),
            cert.r,
            cert.threshold,
            &oracle,
        );
        Ok((case, oracle))
    }
}

pub fn certify(
    image_id: u16,
    true_label: u8,
    predictions: Vec<u8>,
) -> Result<CertifiedSparseInstance, String> {
    let r = predictions.len();
    if usize::from(image_id) >= DATASET_COUNT || true_label >= 10 {
        return Err("MNIST image identifier or true digit is out of range".into());
    }
    if !SUPPORTED_VIEW_COUNTS.contains(&r) {
        return Err(
            "the MNIST circuit uses one of the supported R values: 256, 1024, 4096, 16384, 65536, 262144, 1048576, or a power of two up to 2^30"
                .into(),
        );
    }
    let mut vote_counts = [0usize; 10];
    for &label in &predictions {
        if label >= 10 {
            return Err("MNIST table predictions must be digits 0 through 9".into());
        }
        vote_counts[usize::from(label)] += 1;
    }
    let threshold = r / 2 + 1;
    let flip_distances = vote_counts.map(|votes| {
        if votes >= threshold {
            votes - threshold + 1
        } else {
            threshold - votes
        }
    });
    let min_flip_distance = *flip_distances.iter().min().expect("ten labels");
    let certified_flips = min_flip_distance - 1;
    let accepted_label = vote_counts
        .iter()
        .position(|&votes| votes >= threshold)
        .map(|y| y as u8);
    let correctly_accepted = accepted_label == Some(true_label);
    let positive_whole_circuit = certified_flips > 0;
    let strong_whole_circuit = certified_flips >= r / 8;
    Ok(CertifiedSparseInstance {
        data: CertifiedSparseData {
            image_id,
            true_label,
            certificate: SparseCertificate {
                r,
                threshold,
                vote_counts,
                flip_distances,
                min_flip_distance,
                certified_flips,
                accepted_label,
                correctly_accepted,
                positive_whole_circuit,
                strong_whole_circuit,
                strong_correct: strong_whole_circuit && correctly_accepted,
            },
            predictions,
        },
    })
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionCriterion {
    pub minimum_certified_flips: usize,
    pub require_correct: bool,
    pub ordering: String,
    pub selection: String,
    pub image_ids: Vec<u16>,
    pub strong_radius: usize,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct ToolSourceHashes {
    #[serde(rename = "common.py")]
    pub common: String,
    #[serde(rename = "prepare_sparse.py")]
    pub prepare_sparse: String,
    #[serde(rename = "train_sparse_judge.py")]
    pub train_sparse_judge: String,
    #[serde(rename = "survey_sparse.py")]
    pub survey_sparse: String,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionSources {
    pub test_images_sha256: String,
    pub test_labels_sha256: String,
    pub checkpoint_sha256: String,
    pub judge_model_hash: String,
    pub tool_source_sha256: ToolSourceHashes,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelectedImage {
    pub image_id: u16,
    pub true_label: u8,
    pub accepted_label: u8,
    pub vote_counts: [usize; 10],
    pub flip_distances: [usize; 10],
    pub min_flip_distance: usize,
    pub certified_flips: usize,
    pub table: String,
    pub table_sha256: String,
    pub mask_registry_sha256: String,
    pub image_sha256: String,
    pub image_pixels: Vec<u8>,
    pub correctly_accepted: bool,
    pub positive_whole_circuit: bool,
    pub strong_whole_circuit: bool,
    pub strong_correct: bool,
    pub masks: Vec<[u16; 6]>,
    pub canonical_replay_checked: bool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct SelectionManifest {
    pub schema: String,
    pub circuit: String,
    pub dataset_count: usize,
    pub r: usize,
    pub threshold: usize,
    pub k: usize,
    pub global_seed: u64,
    pub criterion: SelectionCriterion,
    pub sources: SelectionSources,
    pub selected_count: usize,
    pub images: Vec<SelectedImage>,
}

pub struct SelectedSurvey {
    data: SelectedSurveyData,
}

pub struct SelectedSurveyData {
    pub manifest: SelectionManifest,
    pub manifest_sha256: String,
    pub instances: Vec<CertifiedSparseInstance>,
}

impl std::ops::Deref for SelectedSurvey {
    type Target = SelectedSurveyData;
    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

pub fn load_selected(path: impl AsRef<Path>) -> Result<SelectedSurvey, String> {
    let mut bytes = Vec::new();
    std::fs::File::open(path.as_ref())
        .map_err(|e| format!("read selected MNIST artifact: {e}"))?
        .take(MAX_SELECTION_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| format!("read selected MNIST artifact: {e}"))?;
    if bytes.len() as u64 > MAX_SELECTION_BYTES {
        return Err("selected MNIST artifact exceeds its 1 TiB resource bound".into());
    }
    parse_selected(&bytes)
}

pub fn load_selection(
    path: impl AsRef<Path>,
    expected_sha256: Option<&str>,
) -> Result<SelectedSurvey, String> {
    let selected = load_selected(path)?;
    if expected_sha256.is_some_and(|expected| expected != selected.manifest_sha256) {
        return Err("selected MNIST artifact differs from the caller-pinned SHA-256".into());
    }
    Ok(selected)
}

pub fn parse_selected(bytes: &[u8]) -> Result<SelectedSurvey, String> {
    if bytes.len() as u64 > MAX_SELECTION_BYTES {
        return Err("selected MNIST artifact exceeds its 1 TiB resource bound".into());
    }
    let manifest: SelectionManifest =
        serde_json::from_slice(bytes).map_err(|e| format!("selected MNIST schema: {e}"))?;
    if manifest.schema != "mnist_sparse_selection_v1"
        || manifest.circuit != "sparse_mnist_majority_v1"
        || manifest.dataset_count != DATASET_COUNT
        || !SUPPORTED_VIEW_COUNTS.contains(&manifest.r)
        || manifest.threshold != manifest.r / 2 + 1
        || manifest.k != 6
        || manifest.global_seed != 20_260_827
    {
        return Err("selected MNIST artifact does not use the original circuit parameters".into());
    }
    if manifest.selected_count == 0
        || manifest.selected_count > DATASET_COUNT
        || manifest.images.len() != manifest.selected_count
        || manifest.criterion.minimum_certified_flips < 1
        || manifest.criterion.minimum_certified_flips >= manifest.r / 2
        || manifest.criterion.strong_radius != manifest.r / 8
        || !manifest.criterion.require_correct
        || manifest.criterion.ordering != "ascending_test_image_id"
        || manifest.criterion.selection != "fixed_image_ids"
        || manifest.criterion.image_ids.len() != manifest.selected_count
    {
        return Err("selected MNIST count or fixed-image selection criterion is invalid".into());
    }
    let sources = &manifest.sources;
    if sources.test_images_sha256 != TEST_IMAGES_SHA256
        || sources.test_labels_sha256 != TEST_LABELS_SHA256
        || sources.checkpoint_sha256 != CHECKPOINT_SHA256
        || sources.judge_model_hash != JUDGE_MODEL_HASH
    {
        return Err(
            "selected MNIST source identity differs from the original dataset or judge".into(),
        );
    }
    let tools = &sources.tool_source_sha256;
    if [
        &tools.common,
        &tools.prepare_sparse,
        &tools.train_sparse_judge,
        &tools.survey_sparse,
    ]
    .iter()
    .any(|hash| !hash_is_valid(hash))
    {
        return Err("selected MNIST tool identity is not a canonical SHA-256".into());
    }
    let mut instances = Vec::with_capacity(manifest.images.len());
    let mut previous_id = None;
    for (index, row) in manifest.images.iter().enumerate() {
        if previous_id.is_some_and(|previous| row.image_id <= previous)
            || manifest.criterion.image_ids[index] != row.image_id
        {
            return Err(
                "selected MNIST image identifiers must be unique, ascending and the configured fixed identifiers".into(),
            );
        }
        previous_id = Some(row.image_id);
        if row.table.len() != manifest.r || !row.table.bytes().all(|b| b.is_ascii_digit()) {
            return Err("selected MNIST table must contain exactly R ASCII digits".into());
        }
        let instance = certify(
            row.image_id,
            row.true_label,
            row.table.bytes().map(|b| b - b'0').collect(),
        )?;
        let cert = &instance.certificate;
        if row.vote_counts != cert.vote_counts
            || row.flip_distances != cert.flip_distances
            || row.min_flip_distance != cert.min_flip_distance
            || row.certified_flips != cert.certified_flips
            || Some(row.accepted_label) != cert.accepted_label
            || row.correctly_accepted != cert.correctly_accepted
            || row.positive_whole_circuit != cert.positive_whole_circuit
            || row.strong_whole_circuit != cert.strong_whole_circuit
            || row.strong_correct != cert.strong_correct
            || !cert.correctly_accepted
            || cert.certified_flips < manifest.criterion.minimum_certified_flips
            || !row.canonical_replay_checked
        {
            return Err(format!(
                "selected MNIST image {} failed table certificate replay or selection",
                row.image_id
            ));
        }
        let mut table_bytes = row.table.as_bytes().to_vec();
        table_bytes.push(b'\n');
        if sha256(&table_bytes) != row.table_sha256
            || row.image_pixels.len() != 784
            || sha256(&row.image_pixels) != row.image_sha256
        {
            return Err("selected MNIST table or image identity is invalid".into());
        }
        if row.masks.len() != manifest.r {
            return Err("selected MNIST mask registry must contain exactly R rows".into());
        }
        let mut seen = HashSet::with_capacity(manifest.r);
        let mut mask_bytes = String::new();
        for mask in &row.masks {
            if mask[5] >= 784
                || !mask.windows(2).all(|pair| pair[0] < pair[1])
                || !seen.insert(*mask)
                || mask
                    .iter()
                    .any(|&coordinate| row.image_pixels[usize::from(coordinate)] == 0)
            {
                return Err("selected MNIST masks must be unique sorted six-pixel subsets".into());
            }
            for (i, coordinate) in mask.iter().enumerate() {
                if i != 0 {
                    mask_bytes.push(' ');
                }
                mask_bytes.push_str(&coordinate.to_string());
            }
            mask_bytes.push('\n');
        }
        if sha256(mask_bytes.as_bytes()) != row.mask_registry_sha256 {
            return Err("selected MNIST mask registry identity is invalid".into());
        }
        instances.push(instance);
    }
    Ok(SelectedSurvey {
        data: SelectedSurveyData {
            manifest,
            manifest_sha256: sha256(bytes),
            instances,
        },
    })
}
