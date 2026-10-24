use rmip_field::{BinaryField, OpeningField};
use rmip_field::{F128, F64};
use rmip_packed::tree::{bit_claim_at, eval_at, Tree};
use rmip_packed::PackedField;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 11
}

fn answers(views: usize, seed: u64) -> Vec<bool> {
    let mut s = seed;
    (0..views).map(|_| lcg(&mut s) % 100 < 94).collect()
}

fn check_tree<F: PackedField>(views: usize, v_log: usize, c_log: usize) {
    let tree = Tree::new(views, v_log, c_log);
    let ans = answers(views, 0x11 + views as u64);
    let w = tree.witness::<F>(&ans);
    assert_eq!(w.len(), tree.levels.len());
    for (l, (level, lw)) in tree.levels.iter().zip(&w).enumerate() {
        let r1cs = level.r1cs();
        assert!(r1cs.satisfies(&lw.z), "level {l} satisfies its R1CS");
        assert_eq!(
            r1cs.apply(&r1cs.a0, &lw.z).words,
            lw.a.words,
            "level {l}: a = A z"
        );
        assert_eq!(
            r1cs.apply(&r1cs.b0, &lw.z).words,
            lw.b.words,
            "level {l}: b = B z"
        );
    }
    let expected = ans.iter().filter(|&&b| b).count() as u64;
    assert_eq!(tree.count(w.last().unwrap()), expected);
    assert_eq!(tree.levels.last().unwrap().n_log, 0);
    let (offsets, total_log) = tree.layout(F::BITS.trailing_zeros() as usize);
    println!(
        "views {views} at {} bits: levels {:?}, offsets {offsets:?}, total 2^{total_log} words",
        F::BITS,
        tree.levels
            .iter()
            .map(|l| (l.block.k(), 1usize << l.n_log))
            .collect::<Vec<_>>()
    );
}

#[test]
fn the_tree_counts() {
    for views in [64usize, 128, 4096, 1 << 14] {
        check_tree::<F64>(views, 6, 6);
        check_tree::<F128>(views, 6, 6);
    }
    check_tree::<F64>(1 << 16, 6, 3);
}

#[test]
fn copy_points_agree() {
    let mut s = 0x21_u64;
    for views in [4096usize, 1 << 14] {
        let tree = Tree::new(views, 6, 6);
        let ans = answers(views, 0x5);
        let w = tree.witness::<F64>(&ans);
        for level in 1..tree.levels.len() {
            let rho: Vec<F64> = (0..tree.copy_coins(level))
                .map(|_| F64::from_bits(u128::from(lcg(&mut s)) << 20 ^ u128::from(lcg(&mut s))))
                .collect();
            let (input, output) = tree.copy_points::<F64>(level, &rho);
            let vi = eval_at(&w[level].z, &input);
            let vo = eval_at(&w[level - 1].z, &output);
            assert_eq!(vi, vo, "link into level {level}");
            let mut tampered = w[level].z.clone();
            let lv = &tree.levels[level];
            let bit = lv.block.in_base + 3 * lv.block.in_stride + 1;
            let word = bit / 64;
            tampered.words[word] =
                F64::from_bits(tampered.words[word].to_bits() ^ (1 << (bit % 64)));
            assert_ne!(eval_at(&tampered, &input), vo, "a flipped input is seen");
        }
    }
}

#[test]
fn index_helpers_find_the_bits() {
    let views = 4096;
    let tree = Tree::new(views, 6, 6);
    let ans = answers(views, 0x9);
    let w = tree.witness::<F128>(&ans);
    for x in [0usize, 1, 63, 64, 1000, 4095] {
        assert_eq!(w[0].z.bit(tree.answer_bit(x)), ans[x], "view {x}");
        let bit = tree.answer_bit(x);
        let word = w[0].z.words[bit / 128];
        let point: Vec<F128> = (0..tree.levels[0].m())
            .map(|b| F128::from_bool(((bit / 128) >> b.saturating_sub(7)) & 1 == 1 && b >= 7))
            .collect();
        let claim = bit_claim_at(&point, F128::ZERO);
        let _ = claim;
        assert_eq!((word.to_bits() >> (bit % 128)) & 1 == 1, ans[x]);
    }
    let top = w.last().unwrap();
    let word = top.z.words[tree.count_bit() / 128];
    assert_eq!(
        tree.count_from_word(word.to_bits(), 128),
        ans.iter().filter(|&&b| b).count() as u64
    );
}
