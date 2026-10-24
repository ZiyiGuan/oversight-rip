use rmip_frontend::bits::BitVec;
use rmip_frontend::deoracle;
use rmip_frontend::layout::TraceLayout;
use rmip_frontend::mnist::{
    sparse_query_bytes, SparseMajorityFamily, SparseTableOracle, LABEL_BITS,
};
use rmip_frontend::oracle::Oracle;
use rmip_frontend::r1cs::{self, R1cs, R1csConfig};
use rmip_frontend::trace;

fn r1cs_satisfied(sys: &R1cs, z: &[bool]) -> bool {
    let mul = |m: &rmip_frontend::SparseF2| {
        let mut v = vec![false; m.rows];
        for &(r, c) in &m.nz {
            if z[c] {
                v[r] ^= true;
            }
        }
        v
    };
    let (a, b, c) = (mul(&sys.a), mul(&sys.b), mul(&sys.c));
    (0..a.len()).all(|i| (a[i] && b[i]) == c[i])
}

fn label_bits(y: usize) -> BitVec {
    let mut x = BitVec::zeros(LABEL_BITS);
    for k in 0..LABEL_BITS {
        x.set(k, (y >> k) & 1 == 1);
    }
    x
}

#[test]
fn tiny_sparse_majority_votes_and_flip_distances() {
    let table: Vec<u8> = vec![5, 5, 5, 5, 5, 6, 8];
    let fam = SparseMajorityFamily::new(9, 7);
    assert_eq!(fam.threshold, 4, "the default threshold is floor(R/2)+1");
    let circuit = fam.build();
    assert_eq!(circuit.ell(), 7);
    assert!(circuit.b_widths().iter().all(|&w| w == 64));
    let (dc, sites) = deoracle::deoraclize(&circuit);
    let lay = TraceLayout::new(&dc);
    let sys = r1cs::emit(&dc, &lay, &R1csConfig::default());
    let oracle = SparseTableOracle {
        image_id: 9,
        predicted: table.clone(),
    };

    for y in 0..16usize {
        let x = label_bits(y);
        let s_y = table.iter().filter(|&&p| usize::from(p) == y).count();
        let want = y < 10 && s_y >= 4;
        let (got, answers) = circuit.eval_with_answers(x.as_slice(), &oracle);
        assert_eq!(got, want, "label {y}: S_y = {s_y}");
        assert_eq!(answers.len(), 7, "full query plan on every label");
        let z = trace::witness_trace(&dc, &lay, x.as_slice(), answers.as_slice());
        assert_eq!(r1cs_satisfied(&sys, z.as_slice()), want);

        if y >= 10 {
            continue;
        }
        let d_y = if s_y >= 4 { s_y - 4 + 1 } else { 4 - s_y };
        let flip = |count: usize| -> bool {
            let mut a: Vec<bool> = (0..7).map(|r| answers.get(r)).collect();
            let target = want;
            let mut left = count;
            for v in a.iter_mut() {
                if left > 0 && *v == target {
                    *v = !*v;
                    left -= 1;
                }
            }
            assert_eq!(left, 0, "not enough answers to flip");
            dc.eval(x.as_slice(), &a)
        };
        assert_eq!(
            flip(d_y - 1),
            want,
            "label {y}: {} flips must not move the output",
            d_y - 1
        );
        assert_eq!(
            flip(d_y),
            !want,
            "label {y}: {d_y} flips must move the output"
        );
    }
    let _ = sites;
}

#[test]
fn tiny_sparse_queries_are_the_canonical_encoding() {
    let fam = SparseMajorityFamily::new(0xBEEF, 7);
    let circuit = fam.build();
    let (dc, sites) = deoracle::deoraclize(&circuit);
    for (y, y0) in [(3usize, 3u8), (12, 0)] {
        let x = label_bits(y);
        let a = BitVec::zeros(7);
        for r in 0..7 {
            let u = dc.query_bundle(&sites, r, x.as_slice(), a.as_slice());
            let bytes = sparse_query_bytes(0xBEEF, r as u32, y0);
            let want: Vec<bool> = bytes
                .iter()
                .flat_map(|&b| (0..8).map(move |k| (b >> (7 - k)) & 1 == 1))
                .collect();
            assert_eq!(u.as_slice(), &want[..], "label {y} occurrence {r}");
        }
    }
}

#[test]
fn the_table_oracle_is_deterministic_and_total() {
    let oracle = SparseTableOracle {
        image_id: 9,
        predicted: vec![5, 5, 5, 5, 5, 6, 8],
    };
    let bits = |bytes: [u8; 8]| -> BitVec {
        BitVec::from_bools(
            bytes
                .iter()
                .flat_map(|&b| (0..8).map(move |k| (b >> (7 - k)) & 1 == 1))
                .collect(),
        )
    };
    let hit = bits(sparse_query_bytes(9, 5, 6));
    let miss = bits(sparse_query_bytes(9, 5, 7));
    for _ in 0..8 {
        assert!(oracle.query(hit.as_slice()), "view 5 predicts 6");
        assert!(!oracle.query(miss.as_slice()), "view 5 does not predict 7");
    }
    assert!(!oracle.query(&[]), "the empty query is answered, falsely");
    assert!(
        !oracle.query(&hit.as_slice()[..63]),
        "a short query is answered"
    );
    let mut wrong_tag = bits(sparse_query_bytes(9, 5, 6));
    wrong_tag.set(7, !wrong_tag.get(7));
    assert!(
        !oracle.query(wrong_tag.as_slice()),
        "a foreign tag is answered"
    );
    assert!(
        !oracle.query(bits(sparse_query_bytes(10, 5, 6)).as_slice()),
        "a foreign image is answered"
    );
    assert!(
        !oracle.query(bits(sparse_query_bytes(9, 7, 6)).as_slice()),
        "an out-of-range view is answered"
    );
}
