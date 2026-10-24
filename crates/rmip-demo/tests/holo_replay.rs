use rmip_iop::mateval::HoloMatEval;
use rmip_iop::transcript::SeededChallenger;
use std::time::{Duration, Instant};

const REPLAYS: usize = 9;

fn replay_one(c: &rmip_demo::harness::Case) -> (Duration, Duration, Duration) {
    let h = c
        .holo
        .as_deref()
        .expect("an embedded view count carries the index");
    let r_row: Vec<rmip_field::F128> = (0..c.sys.mu)
        .map(|i| rmip_field::F128::new(i as u128 * 7919 + 5))
        .collect();
    let r_col: Vec<rmip_field::F128> = (0..c.layout.nu)
        .map(|i| rmip_field::F128::new(i as u128 * 104_729 + 11))
        .collect();
    let mut pch = SeededChallenger::new(77);
    let (_values, frames) = h.prove_frames(&c.sys.a, &c.sys.b, &c.sys.c, &r_row, &r_col, &mut pch);
    let mut ds = Vec::with_capacity(REPLAYS);
    for _ in 0..REPLAYS {
        let mut vch = SeededChallenger::new(77);
        let mut it = frames.iter();
        let mut next = || it.next().cloned().unwrap_or_default();
        let t0 = Instant::now();
        let out = h.verify_frames(&mut next, &r_row, &r_col, &mut vch);
        let d = t0.elapsed();
        assert!(out.is_ok(), "synthetic holographic verify rejected");
        ds.push(d);
    }
    ds.sort();
    (ds[0], ds[REPLAYS / 2], ds[REPLAYS - 1])
}

#[test]
#[ignore = "measurement"]
fn holographic_verify_replay_on_the_sweep_bundles() {
    for r in [4096usize, 16384] {
        let path = format!(
            "{}/../../benchmarks/mnist-test/artifacts/R{r}/selected.json",
            env!("CARGO_MANIFEST_DIR")
        );
        if !std::path::Path::new(&path).exists() {
            println!("R={r}: bundle absent (gitignored above 1,024 views), skipped");
            continue;
        }
        let sel = rmip_demo::mnist_survey::load_selected(&path).expect("bundle parses");
        for img in [0usize, 25, 49] {
            let (c, _oracle) = sel.instances[img]
                .build_case("sparse-mnist")
                .expect("a certified image builds");
            let (lo, med, hi) = replay_one(&c);
            println!(
                "R={r} image#{img} mu={} nu={} verify_frames best {lo:?} median {med:?} worst {hi:?}",
                c.sys.mu, c.layout.nu
            );
        }
    }
}
