use rmip_field::{BinaryField, F128};
use rmip_frontend::mnist::{MnistInstance, SparseTableOracle};
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, loc, r1cs, Instance};
use rmip_iop::backend::Claim;
use rmip_iop::driver::{boolean_point, committed_trace, execute, Cheat, Config, Statement};
use rmip_iop::lincheck::Coefficients;
use rmip_iop::mle::{eq_table, eval_mle};
use rmip_iop::relation_protocol::{
    verify_interactive, HonestRelationProver, RelationError, RelationMessageProver,
};
use rmip_iop::sumcheck::{recover, SumcheckError};
use rmip_iop::transcript::{Entry, RawChallenger, SeededChallenger, Transcript};
use rmip_iop::{batch, booleanity, matmle};

struct ScriptedChallenger {
    scripted: Vec<F128>,
    used: usize,
    tail: SeededChallenger,
    transcript: Transcript,
}

impl ScriptedChallenger {
    fn new(scripted: Vec<F128>, seed: u64) -> Self {
        Self {
            scripted,
            used: 0,
            tail: SeededChallenger::new(seed),
            transcript: Transcript::new(),
        }
    }
}

impl RawChallenger for ScriptedChallenger {
    fn sample(&mut self) -> F128 {
        let c = if self.used < self.scripted.len() {
            let c = self.scripted[self.used];
            self.used += 1;
            c
        } else {
            self.tail.sample()
        };
        self.transcript.record_coin(c);
        c
    }
    fn record_ordinary(&mut self, bytes: &[u8]) {
        self.transcript.record_ordinary(bytes);
    }
    fn post_oracle(&mut self, id: usize) {
        self.transcript.post_oracle(id);
    }
    fn transcript(&self) -> &Transcript {
        &self.transcript
    }
}

const VIEWS: usize = 16;

struct Fx {
    inst: Instance,
    sys: r1cs::R1cs,
    lay: layout::TraceLayout,
    circuit: rmip_frontend::circuit::OracleCircuit,
    dc: deoracle::OrdinaryCircuit,
    oracle: SparseTableOracle,
}

fn mnist() -> Fx {
    let views = MnistInstance::new(VIEWS);
    let circuit = views.build();
    let oracle = views.oracle();
    let (dc, sites) = deoracle::deoraclize(&circuit);
    let lay = layout::TraceLayout::new(&dc);
    let sys = r1cs::emit(&dc, &lay, &R1csConfig::default());
    let l = loc::build(&sites, &lay);
    let x = views.yes();
    assert!(circuit.eval(x.as_slice(), &oracle));
    let inst = Instance {
        layout: lay.clone(),
        loc: rmip_frontend::loc::Loc::from_rows(l),
        ell: views.ell(),
        b_widths: circuit.b_widths(),
        b_max: rmip_frontend::Instance::max_width(&(circuit.b_widths())),
        x,
        eps_rob: views.eps_rob,
    };
    Fx {
        inst,
        sys,
        lay,
        circuit,
        dc,
        oracle,
    }
}

fn st(f: &Fx) -> Statement<'_> {
    Statement {
        inst: &f.inst,
        sys: &f.sys,
        layout: &f.lay,
        circuit: &f.circuit,
        dc: &f.dc,
        holo: None,
    }
}

fn point(nu: usize, k: u128) -> Vec<F128> {
    (0..nu)
        .map(|i| F128::new((k + 1) * 1_000_003 + i as u128 * 7))
        .collect()
}

fn random_table(nu: usize, seed: u128) -> Vec<F128> {
    (0..1usize << nu)
        .map(|i| F128::new(seed.wrapping_mul(i as u128 + 1).wrapping_add(11)))
        .collect()
}

#[test]
fn two_false_claims_cancel_at_exactly_one_coin() {
    let nu = 6;
    let table = random_table(nu, 0x5eed);
    let (p0, p1) = (point(nu, 0), point(nu, 1));
    let (t0, t1) = (eval_mle(&table, &p0), eval_mle(&table, &p1));

    let d0 = F128::new(0x1234_5678_9abc);
    let d1 = F128::new(0xfeed_face_cafe);
    assert_ne!(d0, d1, "equal errors are the separate case below");
    let false_claims = vec![
        Claim {
            point: p0.clone(),
            value: t0.add(d0),
        },
        Claim {
            point: p1.clone(),
            value: t1.add(d1),
        },
    ];

    let fatal = d0.mul(d0.add(d1).inv().expect("d0 != d1"));
    let coeffs = eq_table(&[fatal]);
    assert_eq!(
        coeffs[0].mul(d0).add(coeffs[1].mul(d1)),
        F128::ZERO,
        "the solved coin must annihilate the error pair"
    );

    let mut ch = ScriptedChallenger::new(vec![fatal], 7);
    let reduced = batch::reduce(&false_claims, &table, &mut ch)
        .expect("at the fatal coin the false pair batches to a true claim");
    assert_eq!(
        eval_mle(&table, &reduced.point),
        reduced.value,
        "and the surviving claim is TRUE of the committed table, which is why \
         the opening cannot catch it either"
    );

    let trials = 4_000usize;
    let mut cancellations = 0usize;
    for seed in 0..trials as u64 {
        let mut ch = SeededChallenger::new(seed.wrapping_mul(0x9e37_79b9) | 1);
        match batch::reduce(&false_claims, &table, &mut ch) {
            Err(SumcheckError::RoundSumMismatch(0)) => {}
            Err(other) => panic!("seed {seed}: unexpected fault {other:?}"),
            Ok(_) => cancellations += 1,
        }
    }
    assert_eq!(
        cancellations, 0,
        "the fatal set has one element out of 2^128; {cancellations} hits in \
         {trials} draws is not that"
    );
    assert!(
        batch::error(2, nu) >= 1.0 / 2f64.powi(128),
        "the declared error must cover one fatal coin out of the field"
    );
}

#[test]
fn equal_errors_on_two_claims_never_cancel() {
    let nu = 6;
    let table = random_table(nu, 0xc0de);
    let d = F128::new(0x0dd_ba11);
    let claims: Vec<Claim<F128>> = (0..2u128)
        .map(|k| {
            let p = point(nu, k);
            Claim {
                value: eval_mle(&table, &p).add(d),
                point: p,
            }
        })
        .collect();
    for seed in 0..500u64 {
        let mut ch = SeededChallenger::new(seed | 1);
        assert_eq!(
            batch::reduce(&claims, &table, &mut ch),
            Err(SumcheckError::RoundSumMismatch(0)),
            "seed {seed}"
        );
    }
}

#[test]
fn no_index_of_a_truncated_padded_claim_list_absorbs_a_false_claim() {
    let f = mnist();
    let nu = f.lay.nu;
    let table = committed_trace(&st(&f), &f.oracle, &Cheat::default());
    let t = 2;
    let k_total = Config::default().k_rel + t * (f.inst.b_max() + 1);
    let coins = batch::coins_for(k_total);
    assert!(
        (1 << (coins - 1)) < k_total && k_total < (1 << coins),
        "K = {k_total} must sit strictly inside its {coins}-coin tensor so the truncation is real"
    );

    let mut honest: Vec<Claim<F128>> = (0..3u128)
        .map(|k| {
            let p = point(nu, k);
            Claim {
                value: eval_mle(&table, &p),
                point: p,
            }
        })
        .collect();
    let loc_0 = f.inst.loc_row(0);
    assert_eq!(loc_0.len(), f.inst.b_max() + 1);
    let mut distinct = loc_0.clone();
    distinct.sort_unstable();
    distinct.dedup();
    assert!(
        distinct.len() < loc_0.len(),
        "the sampled occurrence must repeat coordinates, or nothing here is a duplicate"
    );
    for &coord in &loc_0 {
        let p = boolean_point(coord, nu);
        honest.push(Claim {
            value: eval_mle(&table, &p),
            point: p,
        });
    }
    let pad = honest[1].clone();
    while honest.len() < k_total {
        honest.push(pad.clone());
    }
    assert_eq!(honest.len(), k_total);

    for bad in 0..k_total {
        let mut claims = honest.clone();
        claims[bad].value = claims[bad].value.add(F128::ONE);
        for seed in 0..24u64 {
            let mut ch = SeededChallenger::new(seed.wrapping_mul(2_654_435_761) | 1);
            assert_eq!(
                batch::reduce(&claims, &table, &mut ch),
                Err(SumcheckError::RoundSumMismatch(0)),
                "a false claim at index {bad} survived at seed {seed}"
            );
        }
    }
}

fn subfield_generator(e: u32) -> F128 {
    let order = (1u128 << e) - 1;
    let exponent = u128::MAX / order;
    for seed in 2u128..200 {
        let v = F128::new(seed).pow(exponent);
        let mut frob = v;
        for _ in 0..e {
            frob = frob.square();
        }
        if frob == v && v != F128::ZERO && v != F128::ONE && v.pow(order) == F128::ONE {
            let proper =
                (1..e).all(|d| !e.is_multiple_of(d) || v.pow((1u128 << d) - 1) != F128::ONE);
            if proper {
                return v;
            }
        }
    }
    panic!("no generator of GF(2^{e}) found");
}

#[test]
fn the_subfield_generators_are_what_they_claim_to_be() {
    let g4 = subfield_generator(2);
    assert_eq!(g4.square().square(), g4, "GF(4): v^4 = v");
    assert_ne!(g4.square(), g4, "GF(4): v is not Boolean");
    let g16 = subfield_generator(4);
    let mut f = g16;
    for _ in 0..4 {
        f = f.square();
    }
    assert_eq!(f, g16, "GF(16): v^16 = v");
    assert_ne!(g16.square(), g16, "GF(16): v is not Boolean");
    assert_ne!(g16.square().square(), g16, "GF(16) element is not in GF(4)");
}

#[test]
fn the_residual_is_the_verifier_s_claim() {
    let nu = 6;
    let honest: Vec<F128> = (0..1usize << nu)
        .map(|i| F128::from_bool(i % 3 == 0))
        .collect();
    let mut bad = honest.clone();
    bad[11] = subfield_generator(2);

    for k in 0..64u128 {
        let tau = point(nu, k);
        for (table, boolean) in [(&honest, true), (&bad, false)] {
            let v_z = eval_mle(table, &tau);
            let (sigma, v_sigma) = booleanity::claim_from_lincheck_endpoint(&tau, v_z);
            let claim_holds = eval_mle(table, &sigma) == v_sigma;
            let residual_zero = booleanity::residual_at(table, &tau) == F128::ZERO;
            assert_eq!(
                claim_holds, residual_zero,
                "the appended claim and the residual must agree (k = {k})"
            );
            assert_eq!(residual_zero, boolean, "k = {k}");
        }
    }
}

#[test]
fn tables_in_gf4_and_gf16_are_caught_by_the_closed_form() {
    let nu = 6;
    let trials = 2_000usize;
    for e in [2u32, 4] {
        let g = subfield_generator(e);
        let table: Vec<F128> = (0..1usize << nu)
            .map(|i| g.pow((i as u128 % ((1u128 << e) - 2)) + 1))
            .collect();
        assert!(
            table.iter().all(|v| *v != F128::ZERO && *v != F128::ONE),
            "the table must be non-Boolean everywhere"
        );
        let mut escapes = 0usize;
        for k in 0..trials as u128 {
            if booleanity::residual_at(&table, &point(nu, k)) == F128::ZERO {
                escapes += 1;
            }
        }
        assert_eq!(
            escapes, 0,
            "GF(2^{e}) table escaped {escapes} times in {trials} draws"
        );
    }
    let declared = nu as f64 / 2f64.powi(128);
    assert!(
        declared < 1.0 / trials as f64,
        "the experiment cannot resolve it"
    );
}

#[test]
fn a_single_non_boolean_entry_is_caught_in_every_region() {
    let f = mnist();
    let lay = &f.lay;
    let nu = lay.nu;
    assert!(lay.n + 1 < lay.p, "this shape must have an inner pad");
    assert!(
        lay.used_end() < lay.n_trace,
        "this shape must have a trailing pad"
    );
    let honest = committed_trace(&st(&f), &f.oracle, &Cheat::default());
    assert!(
        honest.iter().all(|v| *v == F128::ZERO || *v == F128::ONE),
        "the honest trace must be Boolean"
    );

    let regions: [(&str, usize); 6] = [
        ("constant slot", 0),
        ("public input block", 1),
        ("inner pad", lay.n + 1),
        ("answer block", lay.p),
        ("ordinary wires", lay.p + lay.ell),
        ("trailing pad", lay.used_end()),
    ];
    for (name, s) in regions {
        assert!(s < lay.n_trace, "{name}: coordinate {s} is off the trace");
        for e in [2u32, 4] {
            let v = subfield_generator(e);
            let mut bad = honest.clone();
            bad[s] = v;
            let jump = v.square().add(v);
            assert_ne!(jump, F128::ZERO, "{name}: the entry must be non-Boolean");
            let mut escapes = 0usize;
            for k in 0..256u128 {
                let tau = point(nu, k);
                let eq_at = eval_mle(&eq_table(&boolean_point(s, nu)), &tau);
                assert_eq!(
                    booleanity::residual_at(&bad, &tau),
                    jump.mul(eq_at),
                    "{name} in GF(2^{e}): the residual is not the closed form (k = {k})"
                );
                escapes += (booleanity::residual_at(&bad, &tau) == F128::ZERO) as usize;
            }
            assert_eq!(escapes, 0, "{name} in GF(2^{e}) escaped {escapes} times");
        }
    }
}

#[test]
fn a_boolean_but_nonzero_trailing_pad_is_accepted_and_is_inert() {
    let f = mnist();
    let lay = &f.lay;
    assert!(
        lay.used_end() + 1 < lay.n_trace,
        "this shape must have a trailing pad of at least two coordinates"
    );
    let honest = committed_trace(&st(&f), &f.oracle, &Cheat::default());
    let (ha, hb, hc) = (
        matmle::mul_vec(&f.sys.a, &honest),
        matmle::mul_vec(&f.sys.b, &honest),
        matmle::mul_vec(&f.sys.c, &honest),
    );

    for coord in [lay.used_end(), lay.used_end() + 1, lay.n_trace - 1] {
        let cheat = Cheat {
            corrupt_trace_coord: Some(coord),
            ..Default::default()
        };
        let padded = committed_trace(&st(&f), &f.oracle, &cheat);
        assert_ne!(padded[coord], honest[coord], "the cheat must have landed");
        assert!(
            padded.iter().all(|v| *v == F128::ZERO || *v == F128::ONE),
            "still Boolean, so the Booleanity check has nothing to say"
        );

        assert_eq!(matmle::mul_vec(&f.sys.a, &padded), ha, "A z moved");
        assert_eq!(matmle::mul_vec(&f.sys.b, &padded), hb, "B z moved");
        assert_eq!(matmle::mul_vec(&f.sys.c, &padded), hc, "C z moved");
        assert_eq!(padded[..lay.p], honest[..lay.p], "the public subcube moved");
        for entry in f.inst.loc_rows() {
            for &c in &entry {
                assert_eq!(padded[c], honest[c], "a loc coordinate moved");
            }
        }

        for seed in 1..=50u64 {
            let mut backend = rmip_backend_trace::RawTrace::new();
            let mut ch = SeededChallenger::new(seed);
            execute(
                &st(&f),
                &mut backend,
                &f.oracle,
                &Config::default(),
                &cheat,
                &mut ch,
            )
            .unwrap_or_else(|r| {
                panic!("a Boolean pad at {coord} was rejected at seed {seed}: {r:?}")
            });
        }
    }
}

struct PatchedRound<'a> {
    inner: HonestRelationProver<'a>,
    sys: &'a r1cs::R1cs,
    victim: usize,
    running: F128,
    row_point: Vec<F128>,
    bound: Vec<F128>,
    coefficients: Option<Coefficients>,
    zerocheck_endpoint: Vec<F128>,
    last_message: Vec<F128>,
}

impl<'a> PatchedRound<'a> {
    fn new(s: &'a Statement<'a>, z: &[F128], victim: usize) -> Self {
        Self {
            inner: HonestRelationProver::new(s, z),
            sys: s.sys,
            victim,
            running: F128::ZERO,
            row_point: Vec::new(),
            bound: Vec::new(),
            coefficients: None,
            zerocheck_endpoint: Vec::new(),
            last_message: Vec::new(),
        }
    }
}

impl RelationMessageProver for PatchedRound<'_> {
    fn observe_public_input_point(&mut self, r: &[F128]) {
        self.inner.observe_public_input_point(r);
    }
    fn begin_zerocheck(&mut self, tau: &[F128]) {
        self.inner.begin_zerocheck(tau);
    }
    fn zerocheck_round_message(&mut self, round: usize) -> Vec<F128> {
        self.inner.zerocheck_round_message(round)
    }
    fn receive_zerocheck_challenge(&mut self, round: usize, c: F128) {
        self.inner.receive_zerocheck_challenge(round, c);
    }
    fn zerocheck_endpoint(&mut self) -> Vec<F128> {
        let e = self.inner.zerocheck_endpoint();
        self.zerocheck_endpoint = e.clone();
        e
    }
    fn holo_frame(&mut self) -> Vec<F128> {
        self.inner.holo_frame()
    }

    fn begin_lincheck(&mut self, row_point: &[F128], coefficients: Coefficients) {
        self.inner.begin_lincheck(row_point, coefficients);
        self.row_point = row_point.to_vec();
        self.coefficients = Some(coefficients);
        let row = rmip_iop::zerocheck::RowClaims {
            v_a: self.zerocheck_endpoint[0],
            v_b: self.zerocheck_endpoint[1],
            v_c: self.zerocheck_endpoint[2],
        };
        self.running = coefficients.batch(row);
    }

    fn lincheck_round_message(&mut self, round: usize) -> Vec<F128> {
        let mut msg = self.inner.lincheck_round_message(round);
        if round == self.victim {
            msg[0] = msg[0].add(F128::ONE);
        }
        self.last_message = msg.clone();
        msg
    }

    fn receive_lincheck_challenge(&mut self, round: usize, c: F128) {
        self.inner.receive_lincheck_challenge(round, c);
        let full = recover(&self.last_message, self.running);
        self.running = full
            .iter()
            .rev()
            .fold(F128::ZERO, |acc, &a| acc.mul(c).add(a));
        self.bound.push(c);
        let _ = round;
    }

    fn lincheck_endpoint(&mut self) -> Vec<F128> {
        let coefficients = self.coefficients.expect("lincheck phase was not started");
        let (ma, mb, mc) = matmle::eval_all(
            &self.sys.a,
            &self.sys.b,
            &self.sys.c,
            &self.row_point,
            &self.bound,
        );
        let l_at = coefficients
            .lambda_a()
            .mul(ma)
            .add(coefficients.lambda_b().mul(mb))
            .add(coefficients.lambda_c().mul(mc));
        let inv = l_at
            .inv()
            .expect("l_at vanishes only with probability 2^-128");
        vec![self.running.mul(inv)]
    }
}

#[test]
fn a_patched_round_polynomial_passes_the_relation_layer_but_not_the_opening() {
    let f = mnist();
    let s = st(&f);
    let z = committed_trace(&s, &f.oracle, &Cheat::default());

    let mut ch = SeededChallenger::new(21);
    let mut honest = HonestRelationProver::new(&s, &z);
    let (_, v) = verify_interactive(&s, &mut honest, &mut ch);
    assert_eq!(v.verdict, Ok(()), "the honest control must accept");
    assert_eq!(
        eval_mle(&z, &v.claims[0].point),
        v.claims[0].value,
        "the honest lincheck claim must be true of the committed table"
    );

    let nu = f.lay.nu;
    for victim in [0usize, 1, nu / 2, nu - 1] {
        let mut ch = SeededChallenger::new(21);
        let mut p = PatchedRound::new(&s, &z, victim);
        let (_, v) = verify_interactive(&s, &mut p, &mut ch);
        assert_eq!(
            v.verdict,
            Ok(()),
            "victim round {victim}: the relation layer has no round-sum check \
             to fail and must accept"
        );
        let claim = &v.claims[0];
        assert_ne!(
            eval_mle(&z, &claim.point),
            claim.value,
            "victim round {victim}: ... while the claim it leaves behind is FALSE, \
             which is what the joint opening refuses"
        );
    }
}

struct WrongWidth<'a> {
    inner: HonestRelationProver<'a>,
    width: usize,
    phase_is_lincheck: bool,
}

impl WrongWidth<'_> {
    fn resize(&self, mut msg: Vec<F128>) -> Vec<F128> {
        msg.resize(self.width, F128::ONE);
        msg
    }
}

impl RelationMessageProver for WrongWidth<'_> {
    fn observe_public_input_point(&mut self, r: &[F128]) {
        self.inner.observe_public_input_point(r);
    }
    fn begin_zerocheck(&mut self, tau: &[F128]) {
        self.inner.begin_zerocheck(tau);
    }
    fn zerocheck_round_message(&mut self, round: usize) -> Vec<F128> {
        let msg = self.inner.zerocheck_round_message(round);
        if self.phase_is_lincheck {
            msg
        } else {
            self.resize(msg)
        }
    }
    fn receive_zerocheck_challenge(&mut self, round: usize, c: F128) {
        self.inner.receive_zerocheck_challenge(round, c);
    }
    fn zerocheck_endpoint(&mut self) -> Vec<F128> {
        self.inner.zerocheck_endpoint()
    }
    fn holo_frame(&mut self) -> Vec<F128> {
        self.inner.holo_frame()
    }
    fn begin_lincheck(&mut self, row_point: &[F128], coefficients: Coefficients) {
        self.inner.begin_lincheck(row_point, coefficients);
    }
    fn lincheck_round_message(&mut self, round: usize) -> Vec<F128> {
        let msg = self.inner.lincheck_round_message(round);
        if self.phase_is_lincheck {
            self.resize(msg)
        } else {
            msg
        }
    }
    fn receive_lincheck_challenge(&mut self, round: usize, c: F128) {
        self.inner.receive_lincheck_challenge(round, c);
    }
    fn lincheck_endpoint(&mut self) -> Vec<F128> {
        self.inner.lincheck_endpoint()
    }
}

#[test]
fn a_round_message_of_the_wrong_width_is_refused_and_buys_no_coins() {
    let f = mnist();
    let s = st(&f);
    let z = committed_trace(&s, &f.oracle, &Cheat::default());

    let coins = |ch: &SeededChallenger| {
        ch.transcript()
            .entries()
            .iter()
            .filter(|e| matches!(e, Entry::Coin(_)))
            .count()
    };

    let mut ch = SeededChallenger::new(33);
    let mut honest = HonestRelationProver::new(&s, &z);
    let (_, v) = verify_interactive(&s, &mut honest, &mut ch);
    assert_eq!(v.verdict, Ok(()));
    let honest_coins = coins(&ch);
    assert!(honest_coins > 0);

    for phase_is_lincheck in [false, true] {
        for width in [3usize, 1] {
            let mut ch = SeededChallenger::new(33);
            let mut p = WrongWidth {
                inner: HonestRelationProver::new(&s, &z),
                width,
                phase_is_lincheck,
            };
            let (proof, v) = verify_interactive(&s, &mut p, &mut ch);
            let expected = SumcheckError::WrongDegree {
                round: 0,
                expected: 2,
                got: width,
            };
            let want = if phase_is_lincheck {
                RelationError::Lincheck(expected)
            } else {
                RelationError::Zerocheck(expected)
            };
            assert_eq!(
                v.verdict,
                Err(want),
                "width {width}, lincheck = {phase_is_lincheck}"
            );
            let recorded = if phase_is_lincheck {
                &proof.lincheck.round_polys
            } else {
                &proof.zerocheck.round_polys
            };
            assert!(
                recorded.iter().all(|m| m.len() == width),
                "the transcript must hold the message as SUPPLIED"
            );
            assert_eq!(
                coins(&ch),
                honest_coins,
                "a malformed message must not change the coin schedule"
            );
        }
    }
}

#[test]
fn the_batching_coins_come_after_every_claim_is_fixed() {
    let f = mnist();
    let cfg = Config::default();
    let k_total = cfg.k_rel + cfg.t * (f.inst.b_max() + 1);
    let gammas = batch::coins_for(k_total);
    assert!(
        cfg.t >= 2 && gammas >= 1,
        "the shape must make the run visible"
    );

    let mut backend = rmip_backend_trace::RawTrace::new();
    let mut ch = SeededChallenger::new(5);
    let (_e, verdict) = rmip_iop::driver::execute_recording(
        &st(&f),
        &mut backend,
        &f.oracle,
        &cfg,
        &Cheat::default(),
        &mut ch,
    );
    assert_eq!(verdict, Ok(()), "the honest control must accept");

    let entries = ch.transcript().entries();
    let is_ordinary = |e: &Entry| matches!(e, Entry::Ordinary(_));
    let block_start = (0..entries.len())
        .find(|&i| i + cfg.t <= entries.len() && entries[i..i + cfg.t].iter().all(is_ordinary))
        .expect("the t query blocks must appear as t consecutive ordinary entries");
    let after = block_start + cfg.t;
    let run = entries[after..]
        .iter()
        .take_while(|e| matches!(e, Entry::Coin(_)))
        .count();
    assert_eq!(
        run, gammas,
        "the coin run after the query blocks must be exactly the ceil(log2 K) \
         batching coins"
    );
}

#[test]
fn a_live_batching_prover_cannot_swap_claims_once_gamma_is_out() {
    use rmip_iop::batch::{verify_interactive as batch_verify, HonestBatchProver};

    let nu = 5;
    let table = random_table(nu, 0xfa11);
    let honest: Vec<Claim<F128>> = (0..2u128)
        .map(|k| {
            let p = point(nu, k);
            Claim {
                value: eval_mle(&table, &p),
                point: p,
            }
        })
        .collect();
    let mut ch = SeededChallenger::new(9);
    let mut prover = HonestBatchProver::new(&honest, &table);
    let out = batch_verify(&honest, nu, &mut prover, &mut ch);
    assert_eq!(out.verification.verdict, Ok(()));

    let mut swapped = honest.clone();
    swapped[0].value = swapped[0].value.add(F128::ONE);
    let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        let mut ch = SeededChallenger::new(9);
        let mut prover = HonestBatchProver::new(&honest, &table);
        batch_verify(&swapped, nu, &mut prover, &mut ch)
    }));
    assert!(
        result.is_err(),
        "a prover handed a different claim list after gamma must refuse"
    );
}

#[test]
#[ignore = "measurement; run with --release --ignored --nocapture"]
fn batching_cancellation_rate_heavy() {
    let nu = 6;
    let table = random_table(nu, 0x5eed);
    let d0 = F128::new(0x1234_5678_9abc);
    let d1 = F128::new(0xfeed_face_cafe);
    let claims: Vec<Claim<F128>> = [(0u128, d0), (1u128, d1)]
        .into_iter()
        .map(|(k, d)| {
            let p = point(nu, k);
            Claim {
                value: eval_mle(&table, &p).add(d),
                point: p,
            }
        })
        .collect();

    let trials = 200_000usize;
    let mut cancellations = 0usize;
    for seed in 0..trials as u64 {
        let mut ch = SeededChallenger::new(seed.wrapping_mul(0x9e37_79b9_7f4a_7c15) | 1);
        if batch::reduce(&claims, &table, &mut ch).is_ok() {
            cancellations += 1;
        }
    }
    let observed = cancellations as f64 / trials as f64;
    let declared = batch::error(claims.len(), nu);
    let sigma = (declared * (1.0 - declared) / trials as f64).sqrt();
    println!("cancellations {cancellations}/{trials} = {observed:.3e}; declared {declared:.3e}");
    assert!(
        observed <= declared + 5.0 * sigma,
        "observed {observed:.3e} above the declared batching error {declared:.3e}"
    );
}
