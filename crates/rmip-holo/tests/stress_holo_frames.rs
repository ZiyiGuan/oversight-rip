use rmip_demo::harness::Case;
use rmip_field::{BinaryField, F128};
use rmip_holo::index::{preprocess, HoloIndex, ProverIndex};
use rmip_holo::Embedded;
use rmip_iop::mateval::{HoloMatEval, MatEvalFault};
use rmip_iop::transcript::{RawChallenger, SeededChallenger};
use std::sync::OnceLock;

const VIEWS: usize = 4096;
const SEED: u64 = 9;

fn case() -> &'static Case {
    static CASE: OnceLock<Case> = OnceLock::new();
    CASE.get_or_init(|| {
        let (c, _o) = rmip_demo::harness::mnist_case(VIEWS);
        let c = rmip_demo::harness::with_forced_index(c);
        assert!(c.holo.is_some(), "the forced index");
        c
    })
}

fn embedded(c: &Case) -> &Embedded {
    c.holo.as_deref().expect("embedded by the shipped rule")
}

fn points(c: &Case) -> (Vec<F128>, Vec<F128>) {
    let r_row = (0..c.sys.mu)
        .map(|i| F128::new(i as u128 * 7919 + 13))
        .collect();
    let r_col = (0..c.layout.nu)
        .map(|i| F128::new(i as u128 * 104_729 + 29))
        .collect();
    (r_row, r_col)
}

fn honest_frames(
    h: &Embedded,
    c: &Case,
    r_row: &[F128],
    r_col: &[F128],
) -> ([F128; 3], Vec<Vec<F128>>) {
    let mut ch = SeededChallenger::new(SEED);
    h.prove_frames(&c.sys.a, &c.sys.b, &c.sys.c, r_row, r_col, &mut ch)
}

fn scheduled_frames(index: &HoloIndex) -> usize {
    3 * (index.zeta + 2) + index.v_pre + 1
}

fn bend(index: HoloIndex) -> Embedded {
    Embedded {
        index,
        wit: ProverIndex {
            tables: Vec::new(),
            flat: Vec::new(),
        },
    }
}

fn replay_with(
    h: &Embedded,
    next: &mut dyn FnMut() -> Vec<F128>,
    r_row: &[F128],
    r_col: &[F128],
) -> (Result<[F128; 3], MatEvalFault>, usize) {
    let mut ch = SeededChallenger::new(SEED);
    let out = h.verify_frames(next, r_row, r_col, &mut ch);
    (out, ch.transcript().entries().len())
}

fn replay(
    h: &Embedded,
    frames: &[Vec<F128>],
    r_row: &[F128],
    r_col: &[F128],
) -> (Result<[F128; 3], MatEvalFault>, usize) {
    let mut it = frames.iter();
    let mut next = || it.next().cloned().unwrap_or_default();
    replay_with(h, &mut next, r_row, r_col)
}

fn endpoint_product(values: &[F128], r_row: &[F128], r_col: &[F128]) -> F128 {
    let mu = r_row.len();
    values
        .iter()
        .take(mu + r_col.len())
        .enumerate()
        .map(|(k, &v)| {
            let rk = if k < mu { r_row[k] } else { r_col[k - mu] };
            F128::ONE.add(rk).add(v)
        })
        .fold(F128::ONE, |acc, f| acc.mul(f))
}

#[test]
fn every_frame_position_rejects_on_the_same_coin_schedule() {
    let c = case();
    let h = embedded(c);
    let (r_row, r_col) = points(c);
    let (t, frames) = honest_frames(h, c, &r_row, &r_col);

    let want = scheduled_frames(&h.index);
    assert_eq!(
        frames.len(),
        want,
        "the schedule is 3*(zeta+2) + v_pre + 1 frames (zeta = {}, v_pre = {})",
        h.index.zeta,
        h.index.v_pre
    );

    let (verdict, honest_entries) = replay(h, &frames, &r_row, &r_col);
    assert_eq!(
        verdict.expect("honest frames must verify"),
        t,
        "the honest replay recovers the proven values"
    );

    for i in 0..frames.len() {
        assert!(
            !frames[i].is_empty(),
            "frame {i} is empty: nothing to flip, and the sweep would be vacuous there"
        );
        let mut bad = frames.clone();
        bad[i][0] = bad[i][0].add(F128::ONE);
        let (out, entries) = replay(h, &bad, &r_row, &r_col);
        assert!(out.is_err(), "a flipped element in frame {i} must fault");
        assert_eq!(
            entries, honest_entries,
            "frame {i}: rejecting must not move the coin schedule"
        );
    }
}

#[test]
fn structural_frame_attacks_reject_on_the_same_coin_schedule() {
    let c = case();
    let h = embedded(c);
    let (r_row, r_col) = points(c);
    let (_t, frames) = honest_frames(h, c, &r_row, &r_col);
    let (verdict, honest_entries) = replay(h, &frames, &r_row, &r_col);
    assert!(verdict.is_ok(), "honest frames must verify");

    let len = frames.len();
    for at in [0usize, 1, 5, len - 2] {
        let mut bad = frames.clone();
        bad.swap(at, at + 1);
        let (out, entries) = replay(h, &bad, &r_row, &r_col);
        assert!(
            out.is_err(),
            "swapping frames {at} and {} must fault",
            at + 1
        );
        assert_eq!(
            entries, honest_entries,
            "swap at {at}: rejecting must not move the coin schedule"
        );
    }

    let mut bad = frames.clone();
    bad.remove(len / 2);
    let (out, entries) = replay(h, &bad, &r_row, &r_col);
    assert!(out.is_err(), "a mid-schedule drop must fault");
    assert_eq!(
        entries, honest_entries,
        "a drop must not move the coin schedule"
    );

    let mut bad = frames.clone();
    bad.insert(len / 2, vec![F128::ONE; 3]);
    let (out, entries) = replay(h, &bad, &r_row, &r_col);
    assert!(out.is_err(), "a mid-schedule insert must fault");
    assert_eq!(
        entries, honest_entries,
        "an insert must not move the coin schedule"
    );

    let mut extra = frames.clone();
    extra.push(vec![F128::ONE; 4]);
    let (out, entries) = replay(h, &extra, &r_row, &r_col);
    assert!(
        out.is_ok(),
        "an excess TRAILING frame is not read here: verify_frames must accept \
         (embed.rs invariant 5); the relation layer catches the leftovers"
    );
    assert_eq!(
        entries, honest_entries,
        "an unread trailing frame must not move the coin schedule either"
    );
}

#[test]
fn starved_frames_fault_without_panicking() {
    let c = case();
    let h = embedded(c);
    let (r_row, r_col) = points(c);
    let (_t, frames) = honest_frames(h, c, &r_row, &r_col);
    let (verdict, honest_entries) = replay(h, &frames, &r_row, &r_col);
    assert!(verdict.is_ok(), "honest frames must verify");

    let mut empty = Vec::new;
    let (out, entries) = replay_with(h, &mut empty, &r_row, &r_col);
    assert!(out.is_err(), "an always-empty frame source must fault");
    assert_eq!(
        entries, honest_entries,
        "starvation must not move the coin schedule"
    );

    let mut junk = || vec![F128::ONE; 7];
    let (out, entries) = replay_with(h, &mut junk, &r_row, &r_col);
    assert!(out.is_err(), "constant-width garbage must fault");
    assert_eq!(
        entries, honest_entries,
        "constant-width garbage must not move the coin schedule"
    );
}

#[test]
fn a_consistent_endpoint_lie_is_caught_on_the_embedded_path() {
    let c = case();
    let h = embedded(c);
    let (r_row, r_col) = points(c);
    let (_t, frames) = honest_frames(h, c, &r_row, &r_col);
    let (verdict, honest_entries) = replay(h, &frames, &r_row, &r_col);
    assert!(verdict.is_ok(), "honest frames must verify");

    let at = h.index.zeta + 1;
    let lam = rmip_holo::pack::LAMBDA;
    let width = h.index.mu + h.index.nu;
    let honest_values: Vec<F128> = frames[at]
        .chunks(lam)
        .flat_map(|chunk| rmip_holo::pack::unpack(chunk, lam).expect("full orbit chunks"))
        .collect();
    let mut values = honest_values.clone();

    let k = width - 1;
    let old = values[k];
    let lie = old.add(F128::ONE);
    values[k] = lie;
    let rk = r_col[h.index.nu - 1];
    let f_old = F128::ONE.add(rk).add(old);
    let f_new = F128::ONE.add(rk).add(lie);
    let rj = r_row[0];
    let f_j = F128::ONE.add(rj).add(values[0]);
    let target_fj = f_j
        .mul(f_old)
        .mul(f_new.inv().expect("nonzero in general position"));
    values[0] = target_fj.add(F128::ONE).add(rj);

    assert_ne!(
        values[k], honest_values[k],
        "the lie must actually move a recovered slot value"
    );
    assert_eq!(
        endpoint_product(&values, &r_row, &r_col),
        endpoint_product(&honest_values, &r_row, &r_col),
        "the compensation must leave the endpoint product unchanged, or this \
         test pins the product check rather than the opening"
    );

    values.resize(frames[at].len(), F128::ZERO);
    let mut bad = frames.clone();
    bad[at] = values
        .chunks(lam)
        .flat_map(rmip_holo::pack::evals_from_values)
        .collect();
    assert_ne!(bad[at], frames[at], "the repacked frame must differ");

    let (out, entries) = replay(h, &bad, &r_row, &r_col);
    assert!(
        out.is_err(),
        "a product-consistent endpoint lie must die at the joint opening"
    );
    assert_eq!(
        entries, honest_entries,
        "rejecting the endpoint lie must not move the coin schedule"
    );
}

#[test]
fn a_corrupted_codeword_faults_on_the_embedded_path() {
    let c = case();
    let h = embedded(c);
    let (r_row, r_col) = points(c);
    let (_t, frames) = honest_frames(h, c, &r_row, &r_col);
    let (verdict, honest_entries) = replay(h, &frames, &r_row, &r_col);
    assert!(verdict.is_ok(), "honest frames must verify");

    let (mut tampered, _wit) = preprocess(&c.sys.a, &c.sys.b, &c.sys.c, c.sys.mu, c.layout.nu);
    for col in tampered.codeword.iter_mut() {
        col.0[0] = col.0[0].add(F128::ONE);
    }
    let bent = bend(tampered);
    let (out, entries) = replay(&bent, &frames, &r_row, &r_col);
    assert!(
        out.is_err(),
        "a corrupted verifier-owned codeword must fault the embedded replay"
    );
    assert_eq!(
        entries, honest_entries,
        "rejecting a corrupted codeword must not move the coin schedule"
    );
}

#[test]
fn the_verifier_path_never_reads_the_prover_index() {
    let c = case();
    let h = embedded(c);
    let (r_row, r_col) = points(c);
    let (t, frames) = honest_frames(h, c, &r_row, &r_col);

    let (index, _wit) = preprocess(&c.sys.a, &c.sys.b, &c.sys.c, c.sys.mu, c.layout.nu);
    assert!(
        index.is_well_formed(),
        "what preprocess builds must pass its own predicate"
    );
    let witless = bend(index);
    let (out, entries) = replay(&witless, &frames, &r_row, &r_col);
    assert_eq!(
        out.expect("an empty prover index must still accept honest frames"),
        t,
        "the witless verifier recovers the same three values"
    );

    let (_verdict, honest_entries) = replay(h, &frames, &r_row, &r_col);
    assert_eq!(
        entries, honest_entries,
        "and it draws the same coins doing it"
    );
}
