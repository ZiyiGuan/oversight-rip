use rmip_backend_interleaved::{Ligero, Shape};
use rmip_field::F128;
use rmip_frontend::mnist::{MnistInstance, SparseTableOracle};
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, loc, r1cs, Instance};
use rmip_iop::backend::{
    BackendParams, Claim, CommitmentBackend, Coord, FreezeError, LongOracle, QueryPlan, Symbol,
};
use rmip_iop::driver::{Cheat, Config, Execution, Reject, Statement};
use rmip_iop::transcript::{Entry, OracleId, RawChallenger, SeededChallenger, TranscriptPrefix};
use rmip_mip_prover::{execute_recording as mip_execute, Eager, LazyBackend, OracleMeta};

const SMALL_VIEWS: usize = 16;

struct Fx {
    inst: Instance,
    sys: r1cs::R1cs,
    layout: layout::TraceLayout,
    circuit: rmip_frontend::circuit::OracleCircuit,
    dc: deoracle::OrdinaryCircuit,
    oracle: SparseTableOracle,
}

fn fx() -> Fx {
    let m = MnistInstance::new(SMALL_VIEWS);
    let circuit = m.build();
    let oracle = m.oracle();
    let (dc, sites) = deoracle::deoraclize(&circuit);
    let lay = layout::TraceLayout::new(&dc);
    let sys = r1cs::emit(&dc, &lay, &R1csConfig::default());
    let l = loc::build(&sites, &lay);
    let x = m.yes();
    assert!(
        circuit.eval(x.as_slice(), &oracle),
        "the true label must be accepted at {SMALL_VIEWS} views"
    );
    let inst = Instance {
        layout: lay.clone(),
        loc: rmip_frontend::loc::Loc::from_rows(l),
        ell: m.ell(),
        b_widths: circuit.b_widths(),
        b_max: rmip_frontend::Instance::max_width(&(circuit.b_widths())),
        x,
        eps_rob: m.eps_rob,
    };
    Fx {
        inst,
        sys,
        layout: lay,
        circuit,
        dc,
        oracle,
    }
}

fn k_claims(f: &Fx) -> usize {
    3 + Config::default().t * (f.inst.b_max() + 1)
}

fn st(f: &Fx) -> Statement<'_> {
    Statement {
        inst: &f.inst,
        sys: &f.sys,
        layout: &f.layout,
        circuit: &f.circuit,
        dc: &f.dc,
        holo: None,
    }
}

fn shape(f: &Fx) -> Shape {
    Shape::balanced(f.layout.nu, 4, 4.0)
}

fn same_execution(a: &Execution, b: &Execution) {
    assert_eq!(a.occurrences.len(), b.occurrences.len(), "occurrence count");
    for (i, (x, y)) in a.occurrences.iter().zip(&b.occurrences).enumerate() {
        assert_eq!(x.oracle, y.oracle, "occurrence {i}: oracle");
        assert_eq!(x.coord, y.coord, "occurrence {i}: coordinate");
        assert_eq!(x.prefix, y.prefix, "occurrence {i}: legal prefix");
        assert_eq!(x.answer, y.answer, "occurrence {i}: ANSWER");
    }
    assert_eq!(a.oracle_calls, b.oracle_calls, "Q_O per execution");
    assert_eq!(a.claims_submitted, b.claims_submitted, "K");
    assert_eq!(a.pad_claims, b.pad_claims, "padding");
    assert_eq!(a.rounds, b.rounds, "r_IOP");
    assert_eq!(a.coins, b.coins, "coins");
    assert_eq!(a.short_bytes, b.short_bytes, "C_short");
    assert_eq!(a.transcript_before_sampling, b.transcript_before_sampling);
    assert_eq!(a.transcript_at_query_start, b.transcript_at_query_start);
    assert_eq!(a.transcript_at_query_end, b.transcript_at_query_end);
}

fn same_transcript(a: &[Entry], b: &[Entry]) {
    assert_eq!(a.len(), b.len(), "transcript length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x, y, "transcript entry {i}");
    }
}

fn embedded(f: &Fx) -> rmip_holo::Embedded {
    let (index, wit) = rmip_holo::preprocess(&f.sys.a, &f.sys.b, &f.sys.c, f.sys.mu, f.layout.nu);
    rmip_holo::Embedded { index, wit }
}

fn st_holo<'a>(f: &'a Fx, h: &'a rmip_holo::Embedded) -> Statement<'a> {
    Statement {
        holo: Some(h),
        ..st(f)
    }
}

fn cheats() -> Vec<(&'static str, Cheat)> {
    vec![
        ("honest", Cheat::default()),
        (
            "bad_oracle_answer",
            Cheat {
                bad_oracle_answer: Some(0),
                ..Cheat::default()
            },
        ),
        (
            "trace_swap",
            Cheat {
                trace_swap: true,
                ..Cheat::default()
            },
        ),
        (
            "wrong_lincheck_endpoint",
            Cheat {
                wrong_lincheck_endpoint: true,
                ..Cheat::default()
            },
        ),
        (
            "wrong_zerocheck_endpoint",
            Cheat {
                wrong_zerocheck_endpoint: true,
                ..Cheat::default()
            },
        ),
        (
            "non_boolean_answer",
            Cheat {
                non_boolean_answer: Some(0),
                ..Cheat::default()
            },
        ),
    ]
}

fn the_two_paths_agree<B: CommitmentBackend<Field = F128>, O: rmip_frontend::oracle::Oracle>(
    label: &str,
    stmt: &Statement<'_>,
    oracle: &O,
    cfg: &Config,
    cheats: &[(&str, Cheat)],
    seeds: std::ops::RangeInclusive<u64>,
    backend: impl Fn() -> B,
) {
    for (name, cheat) in cheats {
        for seed in seeds.clone() {
            let mut b_iop = backend();
            let mut ch_iop = SeededChallenger::new(seed);
            let (e_iop, v_iop) = rmip_iop::driver::execute_recording(
                stmt,
                &mut b_iop,
                oracle,
                cfg,
                cheat,
                &mut ch_iop,
            );

            let mut b_mip = Eager(backend());
            let mut ch_mip = SeededChallenger::new(seed);
            let (e_mip, v_mip) = mip_execute(stmt, &mut b_mip, oracle, cfg, cheat, &mut ch_mip);

            assert_eq!(v_iop, v_mip, "{label}/{name}/seed {seed}: verdicts differ");
            same_execution(&e_iop, &e_mip);
            same_transcript(ch_iop.transcript().entries(), ch_mip.transcript().entries());
        }
    }
}

#[test]
fn the_mip_prover_and_the_iop_driver_agree() {
    let f = fx();
    the_two_paths_agree(
        "scan/interleaved-rs",
        &st(&f),
        &f.oracle,
        &Config::default(),
        &cheats(),
        1..=12,
        || Ligero::new(shape(&f)),
    );
}

#[test]
fn the_port_agrees_with_the_driver_on_an_embedded_statement() {
    let f = fx();
    let h = embedded(&f);

    let entries = |stmt: &Statement<'_>| {
        let mut b = Eager(Ligero::new(shape(&f)));
        let mut ch = SeededChallenger::new(1);
        let (_, v) = mip_execute(
            stmt,
            &mut b,
            &f.oracle,
            &Config::default(),
            &Cheat::default(),
            &mut ch,
        );
        assert!(v.is_ok(), "the probe run must be honest: {v:?}");
        ch.transcript().entries().len()
    };
    let (scan, embed) = (entries(&st(&f)), entries(&st_holo(&f, &h)));
    assert!(
        embed > scan,
        "the embedded statement recorded {embed} transcript entries against the \
         scan's {scan}: the holographic sub-protocol left no trace, so `holo` \
         is not being read and this sweep is a duplicate of the scan one"
    );

    the_two_paths_agree(
        "embedded/interleaved-rs",
        &st_holo(&f, &h),
        &f.oracle,
        &Config::default(),
        &cheats(),
        1..=12,
        || Ligero::new(shape(&f)),
    );
}

#[test]
fn the_port_agrees_with_the_driver_at_the_shipped_embedding_shape() {
    let (c, o) = rmip_demo::harness::mnist_case(4096);
    let c = rmip_demo::harness::with_forced_index(c);
    let h = c.holo.as_deref().expect("the forced index");
    let stmt = Statement {
        inst: &c.inst,
        sys: &c.sys,
        layout: &c.layout,
        circuit: &c.circuit,
        dc: &c.dc,
        holo: Some(h as &dyn rmip_iop::mateval::HoloMatEval),
    };
    let cfg = Config {
        t: c.t,
        ..Config::default()
    };
    let three: Vec<(&str, Cheat)> = cheats()
        .into_iter()
        .filter(|(n, _)| {
            matches!(
                *n,
                "honest" | "bad_oracle_answer" | "wrong_lincheck_endpoint"
            )
        })
        .collect();
    assert_eq!(three.len(), 3, "the cheat names moved");

    the_two_paths_agree("shipped-embedded", &stmt, &o, &cfg, &three, 1..=1, || {
        Ligero::new(Shape::balanced(c.layout.nu, 4, 4.0))
    });
}

#[test]
fn the_port_agrees_with_the_driver_on_the_raw_trace_and_fold_ladder_backends() {
    use rmip_backend_recursive::{FoldLadder, Shape as LadderShape};
    use rmip_backend_trace::RawTrace;

    let f = fx();
    the_two_paths_agree(
        "scan/raw-trace",
        &st(&f),
        &f.oracle,
        &Config::default(),
        &cheats(),
        1..=4,
        RawTrace::new,
    );
    the_two_paths_agree(
        "scan/fold-ladder",
        &st(&f),
        &f.oracle,
        &Config::default(),
        &cheats(),
        1..=4,
        || FoldLadder::new(LadderShape::balanced(f.layout.nu, 4, 4.0)),
    );
}

#[test]
fn the_port_agrees_with_the_driver_under_public_dummy_padding() {
    let f = fx();
    let q = {
        let mut b = Ligero::new(shape(&f));
        let z = vec![F128::ZERO; 1usize << f.layout.nu];
        b.commit(&z, &mut SeededChallenger::new(1));
        b.params(k_claims(&f)).q_long
    };
    let cfg = Config {
        pad_queries_to: Some(q + 5),
        ..Config::default()
    };

    let mut b = Eager(Ligero::new(shape(&f)));
    let mut ch = SeededChallenger::new(1);
    let (e, v) = mip_execute(&st(&f), &mut b, &f.oracle, &cfg, &Cheat::default(), &mut ch);
    assert!(v.is_ok(), "the padded honest run must accept: {v:?}");
    assert_eq!(
        e.occurrences.len(),
        q + 5,
        "the composite occurrence count must be `Config::audit_query_count`, \
         i.e. the backend's {q} plus five dummies -- otherwise the sweep below \
         compares two UNPADDED executions and says nothing about the tail"
    );

    let h = embedded(&f);
    the_two_paths_agree(
        "padded/interleaved-rs",
        &st(&f),
        &f.oracle,
        &cfg,
        &cheats(),
        1..=4,
        || Ligero::new(shape(&f)),
    );
    the_two_paths_agree(
        "padded+embedded/interleaved-rs",
        &st_holo(&f, &h),
        &f.oracle,
        &cfg,
        &cheats(),
        1..=4,
        || Ligero::new(shape(&f)),
    );
}

#[test]
fn the_lazy_commitment_is_the_eager_one() {
    let f = fx();
    for seed in 1..=25u64 {
        let mut b_eager = Eager(Ligero::new(shape(&f)));
        let mut ch_eager = SeededChallenger::new(seed);
        let (e_eager, v_eager) = mip_execute(
            &st(&f),
            &mut b_eager,
            &f.oracle,
            &Config::default(),
            &Cheat::default(),
            &mut ch_eager,
        );

        let mut b_lazy = Ligero::lazy(shape(&f));
        let mut ch_lazy = SeededChallenger::new(seed);
        let (e_lazy, v_lazy) = mip_execute(
            &st(&f),
            &mut b_lazy,
            &f.oracle,
            &Config::default(),
            &Cheat::default(),
            &mut ch_lazy,
        );

        assert!(v_eager.is_ok(), "seed {seed}: eager rejected: {v_eager:?}");
        assert!(v_lazy.is_ok(), "seed {seed}: LAZY rejected: {v_lazy:?}");
        same_execution(&e_eager, &e_lazy);
        same_transcript(
            ch_eager.transcript().entries(),
            ch_lazy.transcript().entries(),
        );
    }
}

#[test]
#[should_panic(expected = "materialises no codeword")]
fn a_lazy_backend_refuses_to_hand_over_a_materialised_oracle() {
    let f = fx();
    let mut b = Ligero::lazy(shape(&f));
    let mut ch = SeededChallenger::new(1);
    let z = vec![F128::ZERO; 1usize << f.layout.nu];
    b.commit(&z, &mut ch);
    let _ = b.oracles();
}

#[test]
fn oracle_meta_uses_the_posted_logical_length() {
    let oracle = LongOracle::<F128>::new_lazy(91, rmip_iop::backend::Alphabet::Field, 17);
    assert_eq!(OracleMeta::of(&oracle).len, Coord::from(17u64));
}

#[test]
fn laziness_does_not_shrink_the_declared_oracle() {
    let f = fx();
    let k = k_claims(&f);
    let (mut eager, mut lazy) = (Ligero::new(shape(&f)), Ligero::lazy(shape(&f)));
    let z = vec![F128::ZERO; 1usize << f.layout.nu];
    eager.commit(&z, &mut SeededChallenger::new(1));
    lazy.commit(&z, &mut SeededChallenger::new(1));

    let (pe, pl) = (eager.params(k), lazy.params(k));
    assert_eq!(pe.l_long, pl.l_long, "L_long");
    assert_eq!(pe.a_long, pl.a_long, "A_long");
    assert_eq!(pe.q_long, pl.q_long, "q_long");
    assert_eq!(pe.eps_open, pl.eps_open, "eps_open");
    assert_eq!(pe.eps_bind, pl.eps_bind, "eps_bind");
    assert_eq!(pe.alphabet, pl.alphabet, "alphabet");
}

struct MisreportedLazyLen {
    inner: Ligero,
    meta_calls: std::cell::Cell<usize>,
    misreport_after: usize,
    query_outside_posted_domain: bool,
}

struct OneCoordinate(Option<usize>);

impl QueryPlan<F128> for OneCoordinate {
    fn next(&mut self, _prev: Option<&Symbol<F128>>) -> Option<(OracleId, Coord)> {
        self.0.take().map(|coord| {
            (
                rmip_backend_interleaved::CODEWORD_ORACLE,
                Coord::from(coord),
            )
        })
    }
}

impl CommitmentBackend for MisreportedLazyLen {
    type Field = F128;
    type Commitment = <Ligero as CommitmentBackend>::Commitment;
    type Verifier = <Ligero as CommitmentBackend>::Verifier;

    fn commit(&mut self, z: &[Self::Field], ch: &mut dyn RawChallenger) -> Self::Commitment {
        self.inner.commit(z, ch)
    }

    fn open(&mut self, claims: &[Claim<Self::Field>], ch: &mut dyn RawChallenger) {
        self.inner.open(claims, ch);
    }

    fn query_plan(&self, ch: &mut dyn RawChallenger) -> Box<dyn QueryPlan<Self::Field>> {
        if self.query_outside_posted_domain {
            let posted_len = self.inner.oracle_shapes()[0].2;
            Box::new(OneCoordinate(Some(posted_len)))
        } else {
            self.inner.query_plan(ch)
        }
    }

    fn oracles(&self) -> &[LongOracle<Self::Field>] {
        self.inner.oracles()
    }

    fn reconstruct(&self, id: OracleId, h: &TranscriptPrefix, coord: Coord) -> Symbol<Self::Field> {
        self.inner.reconstruct(id, h, coord)
    }

    fn verifier(&self) -> Self::Verifier {
        self.inner.verifier()
    }

    fn public_params(&self, trace_len: usize, claims: usize) -> BackendParams {
        self.inner.public_params(trace_len, claims)
    }

    fn params(&self, claims: usize) -> BackendParams {
        self.inner.params(claims)
    }
}

impl LazyBackend for MisreportedLazyLen {
    fn oracle_meta(&self) -> Vec<OracleMeta> {
        let call = self.meta_calls.get();
        self.meta_calls.set(call + 1);
        self.inner
            .oracle_shapes()
            .into_iter()
            .map(|(id, alphabet, len)| OracleMeta {
                id,
                alphabet,
                len: if call >= self.misreport_after {
                    Coord::from(len)
                        .checked_add_u64(1)
                        .expect("test domain overflow")
                } else {
                    Coord::from(len)
                },
            })
            .collect()
    }

    fn symbol_at(&self, id: OracleId, h: &TranscriptPrefix, coord: Coord) -> Symbol<F128> {
        let (_, alphabet, posted_len) = self.inner.oracle_shapes()[0];
        if coord == Coord::from(posted_len) {
            Symbol(vec![F128::ZERO; alphabet.elements()])
        } else {
            self.inner.reconstruct(id, h, coord)
        }
    }
}

#[test]
fn lazy_logical_length_is_frozen_at_the_posting_event() {
    let f = fx();
    let mut backend = MisreportedLazyLen {
        inner: Ligero::lazy(shape(&f)),
        meta_calls: std::cell::Cell::new(0),
        misreport_after: 0,
        query_outside_posted_domain: false,
    };
    let mut ch = SeededChallenger::new(17);
    let (_, verdict) = mip_execute(
        &st(&f),
        &mut backend,
        &f.oracle,
        &Config::default(),
        &Cheat::default(),
        &mut ch,
    );

    assert!(
        matches!(
            verdict,
            Err(Reject::OracleMutation(FreezeError::Replaced(id)))
                if id == rmip_backend_interleaved::CODEWORD_ORACLE
        ),
        "a post-hoc lazy-domain change was not rejected: {verdict:?}"
    );
}

#[test]
fn lazy_query_bounds_use_the_validated_posting_snapshot_without_a_toctou() {
    let f = fx();
    let posted_len = shape(&f).n_enc;
    let mut backend = MisreportedLazyLen {
        inner: Ligero::lazy(shape(&f)),
        meta_calls: std::cell::Cell::new(0),
        misreport_after: 2,
        query_outside_posted_domain: true,
    };
    let mut ch = SeededChallenger::new(23);
    let (_, verdict) = mip_execute(
        &st(&f),
        &mut backend,
        &f.oracle,
        &Config::default(),
        &Cheat::default(),
        &mut ch,
    );

    assert!(
        matches!(
            verdict,
            Err(Reject::OracleInconsistent { oracle, ref coord })
                if oracle == rmip_backend_interleaved::CODEWORD_ORACLE
                    && **coord == Coord::from(posted_len)
        ),
        "query bounds did not use the posting-time domain: {verdict:?}"
    );
}

#[test]
fn a_rejecting_lazy_execution_still_makes_exactly_q_long_occurrences() {
    let f = fx();
    let q = {
        let mut b = Ligero::new(shape(&f));
        let z = vec![F128::ZERO; 1usize << f.layout.nu];
        b.commit(&z, &mut SeededChallenger::new(1));
        b.params(k_claims(&f)).q_long
    };

    for cheat in [
        Cheat {
            bad_oracle_answer: Some(0),
            ..Cheat::default()
        },
        Cheat {
            trace_swap: true,
            ..Cheat::default()
        },
        Cheat {
            wrong_lincheck_endpoint: true,
            ..Cheat::default()
        },
    ] {
        let mut b = Ligero::lazy(shape(&f));
        let mut ch = SeededChallenger::new(7);
        let (e, v) = mip_execute(
            &st(&f),
            &mut b,
            &f.oracle,
            &Config::default(),
            &cheat,
            &mut ch,
        );
        assert!(v.is_err(), "the cheat should have been caught");
        assert_eq!(
            e.occurrences.len(),
            q,
            "q_long must not depend on the verdict"
        );
        assert!(
            !matches!(v, Err(Reject::QueryCount { .. })),
            "wrong rejection: {v:?}"
        );
    }
}

#[test]
fn a_lazy_checker_answers_exactly_as_an_eager_one() {
    use rmip_mip::{PCheck, SharedAdvice};

    let f = fx();
    let advice = SharedAdvice { seed: 0xA11CE };
    let z: Vec<F128> = {
        let mut ch = SeededChallenger::new(advice.seed);
        let mut b = Ligero::new(shape(&f));
        let z = rmip_iop::driver::committed_trace(&st(&f), &f.oracle, &Cheat::default());
        b.commit(&z, &mut ch);
        z
    };

    let (mut eager, mut lazy) = (Ligero::new(shape(&f)), Ligero::lazy(shape(&f)));
    let mut ch_e = SeededChallenger::new(advice.seed);
    let mut ch_l = SeededChallenger::new(advice.seed);
    eager.commit(&z, &mut ch_e);
    lazy.commit(&z, &mut ch_l);
    let h = ch_e
        .transcript()
        .prefix_at(rmip_backend_interleaved::CODEWORD_ORACLE);
    assert_eq!(
        h,
        ch_l.transcript()
            .prefix_at(rmip_backend_interleaved::CODEWORD_ORACLE),
        "the legal prefix must not depend on materialisation"
    );

    let (pe, pl) = (PCheck::new(eager), PCheck::new(lazy));
    let n_enc = shape(&f).n_enc;
    for coord in 0..n_enc {
        assert_eq!(
            pe.answer(
                rmip_backend_interleaved::CODEWORD_ORACLE,
                &h,
                Coord::from(coord)
            ),
            pl.answer(
                rmip_backend_interleaved::CODEWORD_ORACLE,
                &h,
                Coord::from(coord)
            ),
            "checkers disagree at coordinate {coord} of {n_enc}"
        );
    }

    assert_eq!(
        pe.answer(rmip_iop::driver::DUMMY_ORACLE, &h, Coord::ZERO),
        pl.answer(rmip_iop::driver::DUMMY_ORACLE, &h, Coord::ZERO),
    );
}

struct RewritesAfterPosting {
    inner: Ligero,
    posted: Vec<LongOracle<F128>>,
    target: Coord,
}

impl RewritesAfterPosting {
    fn new(inner: Ligero, target: Coord) -> Self {
        Self {
            inner,
            posted: Vec::new(),
            target,
        }
    }
}

impl CommitmentBackend for RewritesAfterPosting {
    type Field = F128;
    type Commitment = <Ligero as CommitmentBackend>::Commitment;
    type Verifier = <Ligero as CommitmentBackend>::Verifier;

    fn commit(&mut self, z: &[Self::Field], ch: &mut dyn RawChallenger) -> Self::Commitment {
        let c = self.inner.commit(z, ch);
        self.posted = self.inner.oracles().to_vec();
        c
    }

    fn open(&mut self, claims: &[Claim<Self::Field>], ch: &mut dyn RawChallenger) {
        self.inner.open(claims, ch);
        for o in &mut self.posted {
            if o.id != rmip_backend_interleaved::CODEWORD_ORACLE {
                continue;
            }
            let mut symbols: Vec<Symbol<F128>> = o.symbols.to_vec();
            let element =
                &mut symbols[usize::try_from(self.target).expect("coord fits usize")].0[0];
            *element = if *element == F128::ZERO {
                F128::ONE
            } else {
                F128::ZERO
            };
            *o = LongOracle::new(o.id, o.alphabet, symbols);
        }
    }

    fn query_plan(&self, ch: &mut dyn RawChallenger) -> Box<dyn QueryPlan<Self::Field>> {
        self.inner.query_plan(ch)
    }

    fn oracles(&self) -> &[LongOracle<Self::Field>] {
        &self.posted
    }

    fn reconstruct(&self, id: OracleId, h: &TranscriptPrefix, coord: Coord) -> Symbol<Self::Field> {
        match self.posted.iter().find(|o| o.id == id) {
            Some(o) if coord < Coord::from(o.symbols.len()) => {
                o.symbols[usize::try_from(coord).expect("coord fits usize")].clone()
            }
            _ => self.inner.reconstruct(id, h, coord),
        }
    }

    fn verifier(&self) -> Self::Verifier {
        self.inner.verifier()
    }

    fn public_params(&self, trace_len: usize, claims: usize) -> BackendParams {
        self.inner.public_params(trace_len, claims)
    }

    fn params(&self, claims: usize) -> BackendParams {
        self.inner.params(claims)
    }
}

#[test]
fn a_post_commit_symbol_rewrite_is_rejected_by_both_paths() {
    let f = fx();
    const SEED: u64 = 31;

    let target = {
        let mut b = Eager(Ligero::new(shape(&f)));
        let mut ch = SeededChallenger::new(SEED);
        let (e, v) = mip_execute(
            &st(&f),
            &mut b,
            &f.oracle,
            &Config::default(),
            &Cheat::default(),
            &mut ch,
        );
        assert!(v.is_ok(), "the probe run must be honest: {v:?}");
        assert_eq!(
            e.occurrences[0].oracle,
            rmip_backend_interleaved::CODEWORD_ORACLE,
            "the first occurrence should read the codeword oracle"
        );
        e.occurrences[0].coord
    };

    let mut b_iop = RewritesAfterPosting::new(Ligero::new(shape(&f)), target);
    let mut ch_iop = SeededChallenger::new(SEED);
    let (e_iop, v_iop) = rmip_iop::driver::execute_recording(
        &st(&f),
        &mut b_iop,
        &f.oracle,
        &Config::default(),
        &Cheat::default(),
        &mut ch_iop,
    );

    let mut b_mip = Eager(RewritesAfterPosting::new(Ligero::new(shape(&f)), target));
    let mut ch_mip = SeededChallenger::new(SEED);
    let (e_mip, v_mip) = mip_execute(
        &st(&f),
        &mut b_mip,
        &f.oracle,
        &Config::default(),
        &Cheat::default(),
        &mut ch_mip,
    );

    for (who, v) in [("driver", &v_iop), ("port", &v_mip)] {
        assert!(
            matches!(
                v,
                Err(Reject::OracleMutation(FreezeError::Replaced(id)))
                    if *id == rmip_backend_interleaved::CODEWORD_ORACLE
            ),
            "{who} did not reject a post-hoc symbol rewrite: {v:?}"
        );
    }
    same_execution(&e_iop, &e_mip);
    same_transcript(ch_iop.transcript().entries(), ch_mip.transcript().entries());
}
