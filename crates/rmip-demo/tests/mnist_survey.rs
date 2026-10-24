use rmip_demo::mnist_bench::{sparse_instance, sparse_mnist_case};
use rmip_demo::mnist_survey::{
    certify, parse_selected, CHECKPOINT_SHA256, JUDGE_MODEL_HASH, TEST_IMAGES_SHA256,
    TEST_LABELS_SHA256,
};
use rmip_frontend::bits::BitVec;
use rmip_iop::driver::Config;
use serde_json::{json, Value};
use sha2::{Digest, Sha256};

fn sha256(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}

fn label_bits(label: usize) -> BitVec {
    BitVec::from_bools((0..4).map(|bit| label & (1 << bit) != 0).collect())
}

#[test]
fn boundary_distances_are_exact_on_both_sides_of_the_threshold() {
    for (votes, accepted) in [(128, None), (129, Some(0)), (130, Some(0))] {
        let mut predictions = vec![0; votes];
        predictions.extend(vec![1; 256 - votes]);
        let certified = certify(0, 0, predictions).unwrap();
        assert_eq!(certified.certificate.accepted_label, accepted);
        assert_eq!(
            certified.certificate.certified_flips,
            usize::from(votes == 130)
        );
        let family = rmip_frontend::mnist::SparseMajorityFamily::new(0, 256);
        let circuit = family.build();
        let (dc, _) = rmip_frontend::deoracle::deoraclize(&circuit);
        let oracle = rmip_frontend::mnist::SparseTableOracle {
            image_id: 0,
            predicted: certified.predictions().to_vec(),
        };
        for label in 0..10 {
            let input = label_bits(label);
            let (answer, answers) = circuit.eval_with_answers(input.as_slice(), &oracle);
            let distance = certified.certificate.flip_distances[label];
            let flip = |count: usize| {
                let mut changed = answers.as_slice().to_vec();
                let mut remaining = count;
                for bit in &mut changed {
                    if remaining > 0 && *bit == answer {
                        *bit = !*bit;
                        remaining -= 1;
                    }
                }
                assert_eq!(remaining, 0);
                dc.eval(input.as_slice(), &changed)
            };
            assert_eq!(flip(distance - 1), answer);
            assert_eq!(flip(distance), !answer);
        }
        if votes <= 129 {
            assert!(certified.build_case("zero-radius").is_err());
        }
    }
}

#[test]
fn robust_rejection_and_robust_wrong_predictions_are_separate_from_accuracy() {
    let diffuse = certify(1, 0, (0..256).map(|i| (i % 10) as u8).collect()).unwrap();
    assert!(diffuse.certificate.strong_whole_circuit);
    assert!(!diffuse.certificate.strong_correct);
    assert_eq!(diffuse.certificate.accepted_label, None);
    assert!(diffuse.build_case("no-majority").is_err());

    let wrong = certify(2, 7, vec![3; 256]).unwrap();
    assert!(wrong.certificate.positive_whole_circuit);
    assert!(!wrong.certificate.correctly_accepted);
    let (case, oracle) = wrong.build_case("wrong-robust-prediction").unwrap();
    assert_eq!(case.inst.x, label_bits(3));
    assert!(case.circuit.eval(case.inst.x.as_slice(), &oracle));
    assert!(!case.circuit.eval(label_bits(7).as_slice(), &oracle));
}

#[test]
fn runtime_case_matches_original_compiled_mnist_circuit_and_all_radii() {
    for r in [256, 1024, 4096] {
        let original = sparse_instance(r);
        let certified =
            certify(original.image_id, original.true_label as u8, original.table).unwrap();
        assert_eq!(certified.certificate.vote_counts, original.votes);
        assert_eq!(certified.certificate.min_flip_distance, original.d_min);
        assert_eq!(certified.certificate.certified_flips, original.d_min - 1);
        assert_eq!(
            certified.certificate.accepted_label,
            Some(original.true_label as u8)
        );
        if r == 256 {
            let (case, oracle) = certified.build_case("survey").unwrap();
            let (existing, _) = sparse_mnist_case(r, "original");
            assert_eq!(case.layout.nu, existing.layout.nu);
            assert_eq!(case.sys.z, existing.sys.z);
            assert!(case.inst.loc_rows().eq(existing.inst.loc_rows()));
            assert_eq!(case.inst.x, existing.inst.x);
            assert_eq!(case.no_instance_x, existing.no_instance_x);
            assert_eq!(case.honest_trace, existing.honest_trace);
            assert_eq!(case.eps_rob, existing.eps_rob);
            assert_eq!(case.eps_rob.min_bad_set(r), original.d_min);
            assert_eq!(case.t, Config::for_instance(&case.inst, 0.125).t);
            assert_eq!(case.t, existing.t);
            assert_eq!(case.holo.is_some(), existing.holo.is_some());
            assert!(case.circuit.b_widths().iter().all(|&width| width == 64));
            for label in 0..16 {
                assert_eq!(
                    case.circuit.eval(label_bits(label).as_slice(), &oracle),
                    label == original.true_label
                );
            }
        }
    }
}

#[test]
fn invalid_tables_and_identifiers_are_refused_before_case_assembly() {
    assert!(certify(10_000, 0, vec![0; 256]).is_err());
    assert!(certify(0, 10, vec![0; 256]).is_err());
    assert!(certify(0, 0, vec![10; 256]).is_err());
    assert!(certify(0, 0, vec![]).is_err());
    assert!(certify(0, 0, vec![0; 257]).is_err());
    assert!(certify(0, 0, vec![0; 512]).is_err());
}

#[test]
fn every_supported_view_count_certifies_and_builds() {
    for r in rmip_demo::mnist_survey::SUPPORTED_VIEW_COUNTS {
        let mut predictions = vec![7u8; r - r / 4];
        predictions.extend(vec![1u8; r / 4]);
        let certified = certify(0, 7, predictions).unwrap();
        let cert = &certified.certificate;
        assert_eq!(cert.threshold, r / 2 + 1);
        assert_eq!(cert.certified_flips, (r - r / 4) - (r / 2 + 1));
        assert!(cert.strong_correct, "3/4 majority clears R/8 at R = {r}");
        if r <= 4096 {
            let (case, _) = certified.build_case("sweep").unwrap();
            assert_eq!(case.inst.ell, r);
            assert_eq!(case.eps_rob.min_bad_set(r), cert.min_flip_distance);
        }
    }
}

#[test]
fn fixed_image_identifiers_are_checked_against_the_bundle() {
    let valid = selection();
    let mut bad = valid.clone();
    bad["criterion"]["image_ids"] = json!([1]);
    assert!(load(&bad).is_err());
    let mut bad = valid.clone();
    bad["criterion"]["image_ids"] = json!([0, 1]);
    assert!(load(&bad).is_err());
    let mut bad = valid.clone();
    bad["criterion"]["minimum_certified_flips"] = json!(0);
    assert!(load(&bad).is_err());
    let mut bad = valid.clone();
    bad["criterion"]["strong_radius"] = json!(31);
    assert!(load(&bad).is_err());
    let mut bad = valid.clone();
    bad["criterion"]["selection"] = json!("first_qualifying");
    assert!(load(&bad).is_err());
    let mut bad = valid;
    bad["r"] = json!(512);
    bad["threshold"] = json!(257);
    assert!(load(&bad).is_err());
}

fn selection() -> Value {
    selection_at(256)
}

fn fixture_mask(r: usize, i: usize) -> [u16; 6] {
    if r == 256 {
        return [0, 1, 2, 3, 4, i as u16 + 5];
    }
    std::array::from_fn(|k| (100 * k + (i / 100usize.pow(k as u32)) % 100) as u16)
}

fn selection_at(r: usize) -> Value {
    let table = "7".repeat(r);
    let table_bytes = format!("{table}\n");
    let pixels = vec![255u8; 784];
    let masks: Vec<[u16; 6]> = (0..r).map(|i| fixture_mask(r, i)).collect();
    let mut mask_bytes = String::new();
    for mask in &masks {
        mask_bytes.push_str(
            &mask
                .iter()
                .map(u16::to_string)
                .collect::<Vec<_>>()
                .join(" "),
        );
        mask_bytes.push('\n');
    }
    let certified = certify(0, 7, vec![7; r]).unwrap();
    let cert = &certified.certificate;
    json!({
        "schema":"mnist_sparse_selection_v1", "circuit":"sparse_mnist_majority_v1",
        "dataset_count":10000, "r":r, "threshold":r / 2 + 1, "k":6, "global_seed":20260827,
        "criterion":{"minimum_certified_flips":1,"require_correct":true,"ordering":"ascending_test_image_id",
            "selection":"fixed_image_ids","image_ids":[0],"strong_radius":r / 8},
        "sources": {
            "test_images_sha256":TEST_IMAGES_SHA256, "test_labels_sha256":TEST_LABELS_SHA256,
            "checkpoint_sha256":CHECKPOINT_SHA256,"judge_model_hash":JUDGE_MODEL_HASH,
            "tool_source_sha256":{"common.py":"a".repeat(64),"prepare_sparse.py":"b".repeat(64),
                "train_sparse_judge.py":"c".repeat(64),"survey_sparse.py":"d".repeat(64)}
        },
        "selected_count":1,"images":[{
            "image_id":0,"true_label":7,"accepted_label":7,
            "vote_counts":cert.vote_counts,"flip_distances":cert.flip_distances,
            "min_flip_distance":cert.min_flip_distance,"certified_flips":cert.certified_flips,
            "table":table,"table_sha256":sha256(table_bytes.as_bytes()),
            "mask_registry_sha256":sha256(mask_bytes.as_bytes()),"image_sha256":sha256(&pixels),
            "image_pixels":pixels,"correctly_accepted":true,"positive_whole_circuit":true,
            "strong_whole_circuit":true,"strong_correct":true,"masks":masks,
            "canonical_replay_checked":true
        }]
    })
}

fn load(value: &Value) -> Result<rmip_demo::mnist_survey::SelectedSurvey, String> {
    parse_selected(&serde_json::to_vec(value).unwrap())
}

#[test]
fn the_two_largest_view_counts_load() {
    for r in [262_144, 1_048_576] {
        let loaded = load(&selection_at(r)).unwrap();
        assert_eq!(loaded.manifest.r, r);
        assert_eq!(loaded.instances.len(), 1);
        let cert = &loaded.instances[0].certificate;
        assert_eq!(cert.r, r);
        assert_eq!(cert.threshold, r / 2 + 1);
        assert_eq!(cert.certified_flips, r / 2 - 1);
        assert!(cert.strong_correct);
    }
}

#[test]
fn selected_certificate_is_recounted_and_metadata_cannot_inflate_it() {
    let valid = selection();
    let loaded = load(&valid).unwrap();
    assert_eq!(loaded.instances.len(), 1);
    assert_eq!(loaded.instances[0].certificate.certified_flips, 127);
    for key in ["min_flip_distance", "certified_flips"] {
        let mut bad = valid.clone();
        bad["images"][0][key] = json!(1000);
        assert!(load(&bad)
            .err()
            .expect("inflated certificate was accepted")
            .contains("certificate replay"));
    }
    let mut bad = valid.clone();
    bad["images"][0]["vote_counts"][7] = json!(255);
    assert!(load(&bad).is_err());
    let mut bad = valid.clone();
    bad["images"][0]["flip_distances"][0] = json!(1);
    assert!(load(&bad).is_err());
    let mut bad = valid.clone();
    bad["images"][0]["true_label"] = json!(8);
    assert!(load(&bad).is_err());
    let mut bad = valid;
    bad["images"][0]["canonical_replay_checked"] = json!(false);
    assert!(load(&bad).is_err());
}

#[test]
fn strict_schema_source_identity_order_and_artifact_bytes_are_checked() {
    let valid = selection();
    let mut bad = valid.clone();
    bad["extra"] = json!(true);
    assert!(load(&bad).is_err());
    let mut bad = valid.clone();
    bad["sources"]["judge_model_hash"] = json!("0".repeat(64));
    assert!(load(&bad).is_err());
    let mut bad = valid.clone();
    bad["dataset_count"] = json!(9999);
    assert!(load(&bad).is_err());
    let mut bad = valid.clone();
    bad["images"][0]["table"] = json!("7".repeat(255));
    assert!(load(&bad).is_err());
    let mut bad = valid.clone();
    bad["images"][0]["table_sha256"] = json!("0".repeat(64));
    assert!(load(&bad).is_err());
    let mut bad = valid.clone();
    bad["images"][0]["image_pixels"][0] = json!(0);
    assert!(load(&bad).is_err());
    let mut bad = valid.clone();
    bad["selected_count"] = json!(2);
    bad["images"]
        .as_array_mut()
        .unwrap()
        .push(valid["images"][0].clone());
    assert!(load(&bad).is_err());
    let duplicate =
        serde_json::to_string(&valid)
            .unwrap()
            .replacen("\"r\":256", "\"r\":256,\"r\":256", 1);
    assert!(parse_selected(duplicate.as_bytes()).is_err());
}

#[test]
fn malformed_or_zero_pixel_masks_are_refused_even_with_rehashed_bytes() {
    for mask in [
        json!([0, 1, 2, 3, 4, 784]),
        json!([0, 1, 2, 3, 4, 4]),
        json!([0, 1, 2, 4, 3, 5]),
    ] {
        let mut bad = selection();
        bad["images"][0]["masks"][0] = mask;
        assert!(load(&bad).is_err());
    }
    let mut bad = selection();
    bad["images"][0]["masks"][1] = bad["images"][0]["masks"][0].clone();
    assert!(load(&bad).is_err());
    let mut bad = selection();
    let mut pixels = vec![255u8; 784];
    pixels[0] = 0;
    bad["images"][0]["image_pixels"] = json!(pixels);
    bad["images"][0]["image_sha256"] = json!(sha256(&pixels));
    assert!(load(&bad).is_err());
}
