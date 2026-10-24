use rmip_frontend::bits::BitVec;
use rmip_frontend::deoracle::{self, OracleSite, OrdinaryCircuit};
use rmip_frontend::mnist::MnistInstance;
use rmip_frontend::oracle::Oracle;
use rmip_iop::driver::Cheat;

#[derive(Clone, Debug)]
pub struct Prediction {
    pub accept_rate: f64,
    pub derivation: &'static str,
}

#[derive(Clone, Debug)]
pub struct Fixture {
    pub name: &'static str,
    pub cheat: Cheat,
    pub prediction: Prediction,
    pub corrupt_one_query_answer: bool,
}

pub fn honest() -> Fixture {
    Fixture {
        name: "Honest",
        cheat: Cheat::default(),
        prediction: Prediction {
            accept_rate: 1.0,
            derivation: "at least 1-c_open(K); deficit below display precision",
        },
        corrupt_one_query_answer: false,
    }
}

pub fn local_cheat<O: Oracle>(
    dc: &OrdinaryCircuit,
    sites: &[OracleSite],
    x: &BitVec,
    ell: usize,
    t: usize,
    oracle: &O,
) -> Option<Fixture> {
    let (answers, bad) = minimal_bad_answers(dc, sites, x, ell, oracle)?;
    let escape = (1.0 - bad as f64 / ell as f64).powi(t as i32);
    Some(Fixture {
        name: "LocalCheat",
        cheat: Cheat {
            local_cheat_answers: Some(answers),
            ..Default::default()
        },
        prediction: Prediction {
            accept_rate: escape,
            derivation: "(1 - |Bad|/ell)^t, exact for the minimal bad set",
        },
        corrupt_one_query_answer: false,
    })
}

pub fn honest_answers_fixpoint<O: Oracle>(
    dc: &OrdinaryCircuit,
    sites: &[OracleSite],
    x: &BitVec,
    ell: usize,
    oracle: &O,
) -> Option<Vec<bool>> {
    let mut a = vec![false; ell];
    for _ in 0..=ell {
        let w = dc.eval_all_wires(x.as_slice(), &a);
        let next: Vec<bool> = sites
            .iter()
            .map(|s| {
                let u: Vec<bool> = s.query_wires.iter().map(|&q| w[q as usize]).collect();
                oracle.query(&u)
            })
            .collect();
        if next == a {
            return Some(a);
        }
        a = next;
    }
    None
}

pub fn local_cheat_majority<O: Oracle>(
    inst: &MnistInstance,
    dc: &OrdinaryCircuit,
    sites: &[OracleSite],
    x: &BitVec,
    t: usize,
    oracle: &O,
) -> Fixture {
    assert_eq!(x, &inst.no(), "the local cheat lives on the no-instance");
    assert_eq!(sites.len(), inst.ell());
    let a = inst.local_cheat_answers();
    assert!(
        dc.eval(x.as_slice(), &a),
        "the constructed vector satisfies Ĉ"
    );
    let bad = deoracle::bad_set(dc, sites, x.as_slice(), &a, oracle).len();
    assert_eq!(
        bad,
        inst.minimal_bad_set_size(),
        "constructed bad set is T - S_no"
    );
    let escape = (1.0 - bad as f64 / inst.ell() as f64).powi(t as i32);
    Fixture {
        name: "LocalCheat",
        cheat: Cheat {
            local_cheat_answers: Some(a),
            ..Default::default()
        },
        prediction: Prediction {
            accept_rate: escape,
            derivation: "(1 - |Bad|/ell)^t, exact for the minimal bad set",
        },
        corrupt_one_query_answer: false,
    }
}

pub fn minimal_bad_answers<O: Oracle>(
    dc: &OrdinaryCircuit,
    sites: &[OracleSite],
    x: &BitVec,
    ell: usize,
    oracle: &O,
) -> Option<(Vec<bool>, usize)> {
    assert!(
        ell <= 20,
        "exhaustive search is 2^ell; use a targeted construction above 20"
    );
    let mut best: Option<(Vec<bool>, usize)> = None;
    for mask in 0u32..(1u32 << ell) {
        let a: Vec<bool> = (0..ell).map(|j| (mask >> j) & 1 == 1).collect();
        if !dc.eval(x.as_slice(), &a) {
            continue;
        }
        let bad = deoracle::bad_set(dc, sites, x.as_slice(), &a, oracle).len();
        if bad == 0 {
            continue;
        }
        if best.as_ref().is_none_or(|(_, b)| bad < *b) {
            best = Some((a, bad));
        }
    }
    best
}

pub fn oracle_inconsistent(q_long: usize) -> Fixture {
    Fixture {
        name: "OracleInconsistent",
        cheat: Cheat::default(),
        prediction: Prediction {
            accept_rate: 1.0 - 1.0 / q_long as f64,
            derivation: "1 - 1/q_long, the one-occurrence audit",
        },
        corrupt_one_query_answer: true,
    }
}

pub fn oracle_inconsistent_sequential() -> Fixture {
    Fixture {
        name: "OracleInconsistent",
        cheat: Cheat::default(),
        prediction: Prediction {
            accept_rate: 0.0,
            derivation: "0, every occurrence audited in posting order",
        },
        corrupt_one_query_answer: true,
    }
}

pub fn trace_swap() -> Fixture {
    Fixture {
        name: "TraceSwap",
        cheat: Cheat {
            trace_swap: true,
            ..Default::default()
        },
        prediction: Prediction {
            accept_rate: 0.0,
            derivation: "joint opening; deterministic",
        },
        corrupt_one_query_answer: false,
    }
}

pub fn public_input_swap() -> Fixture {
    Fixture {
        name: "PublicInputSwap",
        cheat: Cheat {
            public_input_bits: vec![0, 1],
            ..Default::default()
        },
        prediction: Prediction {
            accept_rate: 0.0,
            derivation: "public-input claim; deterministic",
        },
        corrupt_one_query_answer: false,
    }
}
