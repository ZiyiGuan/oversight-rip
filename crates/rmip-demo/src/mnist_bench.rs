use crate::harness::{with_shipped_embedding, Case};
use rmip_frontend::bits::BitVec;
use rmip_frontend::mnist::{MnistInstance, SparseTableOracle};
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, loc, r1cs, trace, Instance, Ratio};

pub use rmip_frontend::mnist::{parse_digit_row, sparse_instance, RowError, SparseInstance};

#[allow(clippy::too_many_arguments)]
pub(crate) fn assemble(
    family: &'static str,
    circuit: rmip_frontend::circuit::OracleCircuit,
    chain: Option<rmip_frontend::circuit::ChainShape>,
    yes: BitVec,
    no: BitVec,
    eps_rob: Ratio,
    ell: usize,
    majority_threshold: usize,
    oracle: &impl rmip_frontend::oracle::Oracle,
) -> Case {
    let mut circuit = circuit;
    let (mut dc, mut sites) = deoracle::deoraclize(&circuit);
    let lay = layout::TraceLayout::new(&dc);
    let mut sys = r1cs::emit(&dc, &lay, &R1csConfig::default());
    let closed_form = chain.as_ref().map(|shape| {
        std::sync::Arc::new(rmip_iop::closedform::ChainMatEval::new(
            &circuit, &dc, &lay, shape, sys.mu,
        ))
    });
    circuit.gates.shrink_to_fit();
    dc.gates.shrink_to_fit();
    sites.shrink_to_fit();
    let sites: std::sync::Arc<[deoracle::OracleSite]> = sites.into();
    sys.a.nz.shrink_to_fit();
    sys.b.nz.shrink_to_fit();
    sys.c.nz.shrink_to_fit();
    assert!(
        circuit.eval(yes.as_slice(), oracle),
        "the yes-instance must accept"
    );
    assert!(
        !circuit.eval(no.as_slice(), oracle),
        "the no-instance must reject"
    );
    let (_, a) = circuit.eval_with_answers(yes.as_slice(), oracle);
    let z = trace::witness_trace(&dc, &lay, yes.as_slice(), a.as_slice());
    let inst = Instance {
        layout: lay.clone(),
        loc: loc::Loc::from_sites(std::sync::Arc::clone(&sites)),
        ell,
        b_widths: circuit.b_widths(),
        b_max: rmip_frontend::Instance::max_width(&(circuit.b_widths())),
        x: yes,
        eps_rob,
    };
    let t = rmip_iop::driver::Config::for_instance(&inst, 0.125).t;
    with_shipped_embedding(Case {
        family,
        inst,
        sys,
        layout: lay,
        circuit,
        dc,
        sites,
        no_instance_x: no,
        honest_trace: rmip_iop::mle::lift(z.as_slice()),
        t,
        eps_rob,
        majority_threshold,
        holo: None,
        closed_form,
        t_preprocess_verifier: None,
        t_preprocess_prover: None,
        t_preprocess: None,
    })
}

pub fn sparse_mnist_case(r: usize, family: &'static str) -> (Case, SparseTableOracle) {
    let inst = MnistInstance::new(r);
    let (circuit, shape) = inst.build_with_shape();
    let oracle = inst.oracle();
    let case = assemble(
        family,
        circuit,
        Some(shape),
        inst.yes(),
        inst.no(),
        inst.eps_rob,
        inst.ell(),
        inst.threshold,
        &oracle,
    );
    (case, oracle)
}
