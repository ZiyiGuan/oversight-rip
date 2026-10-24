use rmip_demo::fixtures::{honest_answers_fixpoint, local_cheat, local_cheat_majority};
use rmip_demo::harness::mnist_case;
use rmip_frontend::bits::{BitSlice, BitVec};
use rmip_frontend::circuit::{CircuitBuilder, OracleCircuit, WireId};
use rmip_frontend::deoracle;
use rmip_frontend::mnist::MnistInstance;
use rmip_frontend::oracle::Oracle;

const VIEWS: usize = 16;

#[test]
fn the_closed_form_local_cheat_matches_the_exhaustive_search() {
    let (case, oracle) = mnist_case(VIEWS);
    let inst = MnistInstance::new(VIEWS);
    let local = case.clone_for_no_instance();
    let x = &local.inst.x;
    assert_eq!(x, &inst.no(), "the no-instance twin carries the no-label");
    assert!(
        !local.circuit.eval(x.as_slice(), &oracle),
        "the no-label must be rejected"
    );

    let closed = local_cheat_majority(&inst, &local.dc, &local.sites, x, local.t, &oracle);
    let searched = local_cheat(&local.dc, &local.sites, x, local.inst.ell, local.t, &oracle)
        .expect("a no-instance must admit a satisfying answer vector");

    let bad_of = |fx: &rmip_demo::fixtures::Fixture| -> usize {
        let a = fx
            .cheat
            .local_cheat_answers
            .as_ref()
            .expect("a LocalCheat carries answers");
        assert!(local.dc.eval(x.as_slice(), a), "the vector satisfies Ĉ");
        deoracle::bad_set(&local.dc, &local.sites, x.as_slice(), a, &oracle).len()
    };
    let (bad_closed, bad_searched) = (bad_of(&closed), bad_of(&searched));
    assert_eq!(
        bad_closed, bad_searched,
        "the closed form must reach the exhaustive minimum"
    );
    assert_eq!(bad_closed, inst.minimal_bad_set_size(), "|Bad| = T - S_no");
    assert_eq!(
        bad_closed, 9,
        "T = 9 and the no-label has no votes at 16 views"
    );

    assert!(bad_closed >= inst.eps_rob.min_bad_set(inst.ell()));
    assert_eq!(inst.eps_rob.min_bad_set(inst.ell()), inst.d_min);

    assert_eq!(
        closed.prediction.accept_rate,
        searched.prediction.accept_rate
    );
    assert_eq!(local.t, 4);
    assert!((closed.prediction.accept_rate - (7.0 / 16.0f64).powi(4)).abs() < 1e-12);
}

#[test]
#[should_panic(expected = "no-instance")]
fn the_closed_form_local_cheat_refuses_the_yes_instance() {
    let (case, oracle) = mnist_case(VIEWS);
    let inst = MnistInstance::new(VIEWS);
    assert_eq!(case.inst.x, inst.yes());
    let _ = local_cheat_majority(&inst, &case.dc, &case.sites, &case.inst.x, case.t, &oracle);
}

struct XorOracle;

impl Oracle for XorOracle {
    fn query(&self, u: &BitSlice) -> bool {
        u.iter().fold(false, |acc, &b| acc ^ b)
    }
}

fn depth_two_circuit() -> OracleCircuit {
    let mut bld = CircuitBuilder::new(3);
    let xs: Vec<WireId> = (0..3).map(|i| bld.input(i)).collect();
    let a0 = bld.oracle(xs.clone());
    let a1 = bld.oracle(vec![xs[0], xs[1], a0]);
    let a2 = bld.oracle(vec![xs[2], a1]);
    let out = bld.xor(a2, a0);
    bld.finish(out)
}

#[test]
fn the_honest_answers_are_a_fixpoint_at_depth_two() {
    let circuit = depth_two_circuit();
    let (dc, sites) = deoracle::deoraclize(&circuit);
    let x = BitVec::from_bools(vec![true, false, false]);
    let o = XorOracle;

    let one_pass = |a: &Vec<bool>| -> Vec<bool> {
        let w = dc.eval_all_wires(x.as_slice(), a);
        sites
            .iter()
            .map(|s| {
                let u: Vec<bool> = s.query_wires.iter().map(|&q| w[q as usize]).collect();
                o.query(&u)
            })
            .collect()
    };

    let fix = honest_answers_fixpoint(&dc, &sites, &x, 3, &o).expect("a fixpoint exists");
    assert_eq!(
        one_pass(&fix),
        fix,
        "the returned vector must re-derive itself"
    );

    let zero = vec![false; 3];
    let after_one = one_pass(&zero);
    assert_ne!(
        after_one, fix,
        "one pass from the all-zero vector must NOT already be the fixpoint here, \
         or this circuit does not exercise the iteration"
    );
}
