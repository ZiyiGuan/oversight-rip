use rmip_frontend::bits::BitVec;
use rmip_frontend::mnist::{label_bits, MnistInstance};
use rmip_frontend::oracle::Oracle;
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, loc, r1cs, trace};

fn honest_queries(inst: &MnistInstance, y: usize) -> Vec<(Vec<bool>, bool)> {
    let circuit = inst.build();
    let oracle = inst.oracle();
    let (dc, sites) = deoracle::deoraclize(&circuit);
    let x = label_bits(y);
    let (_, a) = circuit.eval_with_answers(x.as_slice(), &oracle);
    let w = dc.eval_all_wires(x.as_slice(), a.as_slice());
    sites
        .iter()
        .map(|s| {
            let u: Vec<bool> = s.query_wires.iter().map(|&q| w[q as usize]).collect();
            (u, a.get(s.occurrence))
        })
        .collect()
}

#[test]
fn the_table_oracle_distinguishes_a_reversed_query() {
    let inst = MnistInstance::new(16);
    let oracle = inst.oracle();
    let queries = honest_queries(&inst, inst.true_label);
    assert_eq!(queries.len(), 16);
    let trues = queries.iter().filter(|(_, a)| *a).count();
    assert_eq!(
        trues, inst.votes[inst.true_label],
        "one true answer per vote"
    );
    assert_eq!(trues, 16, "all sixteen views vote for the seven");

    let mut reversed_differs = 0usize;
    let mut transposed_differs = 0usize;
    for (j, (u, a)) in queries.iter().enumerate() {
        assert_eq!(u.len(), 64, "every sparse-judge query is 64 wires");
        assert_eq!(oracle.query(u), *a, "the honest string answers as recorded");

        let mut rev = u.clone();
        rev.reverse();
        if rev != *u && oracle.query(&rev) != *a {
            reversed_differs += 1;
        }

        let mut swapped = u.clone();
        for k in 0..8 {
            swapped.swap(24 + k, 32 + k);
        }
        if j == 0 {
            assert_eq!(swapped, *u, "mask 0's low bytes are both zero");
        } else if oracle.query(&swapped) != *a {
            transposed_differs += 1;
        }
    }
    assert_eq!(
        reversed_differs, trues,
        "every true answer must flip under reversal, and no false one can"
    );
    assert_eq!(
        transposed_differs,
        trues - 1,
        "every true answer but mask 0's must flip under the byte transposition"
    );
}

#[test]
fn read_block_recovers_the_bundle_in_oracle_order() {
    let inst = MnistInstance::new(256);
    let circuit = inst.build();
    let o = inst.oracle();
    let (dc, sites) = deoracle::deoraclize(&circuit);
    let lay = layout::TraceLayout::new(&dc);
    let l = loc::build(&sites, &lay);
    let ell = inst.ell();

    let mut checked = 0usize;
    let mut trues = 0usize;
    for y in 0..16usize {
        let x = label_bits(y);
        let (_, a) = circuit.eval_with_answers(x.as_slice(), &o);
        let z = trace::witness_trace(&dc, &lay, x.as_slice(), a.as_slice());

        for (j, loc_j) in l.iter().enumerate().take(ell) {
            let (u_read, a_read) = trace::read_block(z.as_slice(), loc_j);
            assert_eq!(
                o.query(u_read.as_slice()),
                a_read,
                "label {y}, occurrence {j}: the block was read in an order the \
                 oracle does not agree with"
            );
            checked += 1;
            trues += usize::from(a_read);
        }
    }
    assert_eq!(l.len(), ell, "loc must carry one row per occurrence");
    assert_eq!(
        checked,
        16 * ell,
        "every occurrence of every label must have been checked"
    );
    let expected_trues: usize = (0..16usize)
        .map(|y| inst.votes[if y < 10 { y } else { 0 }])
        .sum();
    assert_eq!(
        trues, expected_trues,
        "the read answers must carry the votes"
    );
    assert!(
        trues > 0,
        "the check must see true answers, or it is vacuous"
    );
}

#[test]
fn an_honest_execution_is_accepted_under_the_table_oracle() {
    use rmip_backend_trace::RawTrace;
    use rmip_iop::driver::{execute, Cheat, Config, Statement};
    use rmip_iop::transcript::OsChallenger;

    let inst = MnistInstance::new(16);
    let circuit = inst.build();
    let o = inst.oracle();
    let (dc, sites) = deoracle::deoraclize(&circuit);
    let lay = layout::TraceLayout::new(&dc);
    let sys = r1cs::emit(&dc, &lay, &R1csConfig::default());
    let l = loc::build(&sites, &lay);

    let x: BitVec = inst.yes();
    assert!(circuit.eval(x.as_slice(), &o), "the true label accepts");

    let frontend = rmip_frontend::Instance {
        layout: lay.clone(),
        loc: rmip_frontend::loc::Loc::from_rows(l),
        ell: inst.ell(),
        b_widths: circuit.b_widths(),
        b_max: rmip_frontend::Instance::max_width(&(circuit.b_widths())),
        x,
        eps_rob: inst.eps_rob,
    };
    let st = Statement {
        inst: &frontend,
        sys: &sys,
        layout: &lay,
        circuit: &circuit,
        dc: &dc,
        holo: None,
    };
    let cfg = Config::for_instance(&frontend, 0.125);

    for _ in 0..4 {
        let mut backend = RawTrace::new();
        let mut ch = OsChallenger::new();
        execute(&st, &mut backend, &o, &cfg, &Cheat::default(), &mut ch)
            .expect("honest execution rejected under the table oracle");
    }
}
