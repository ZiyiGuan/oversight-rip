use rmip_field::{BinaryField, F128};
use rmip_iop::backend::{
    Alphabet, AuditMode, BackendParams, BooleanBinding, Claim, CommitmentBackend, Coord,
    LongOracle, OpeningVerifier, QueryPlan, Regime, Symbol,
};
use rmip_iop::transcript::{OracleId, RawChallenger, TranscriptPrefix};

const ORACLE: OracleId = 0;
const GHOST: OracleId = 9;

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Mutation {
    None,
    PlanTooShort,
    PlanOutOfRange,
    LyingReconstruct,
    UnannouncedOracle,
    ContentlessAnnouncement,
    EpsOpenDecreasing,
    ParamsChangeAfterCommit,
    NeedsBooleanityRows,
    WrongALong,
    PostOracleWhileSampling,
    SpeakWhileSampling,
    UnsampledPlan,
    OodDoneProperly,
    OodCoinsButNoResponse,
    OodAfterTheCut,
    OodTooFewSamples,
    SpendsCoinsButIgnoresThem,
    RepeatsOneSample,
    LowerHalfBias,
    FakeFullRead,
}

struct Toy {
    mutation: Mutation,
    values: Vec<F128>,
    oracles: Vec<LongOracle<F128>>,
    q: usize,
}

impl Toy {
    fn new(mutation: Mutation) -> Self {
        Self {
            mutation,
            values: Vec::new(),
            oracles: Vec::new(),
            q: 4,
        }
    }
}

struct Picked {
    picked: Vec<usize>,
    next: usize,
}

impl QueryPlan<F128> for Picked {
    fn next(&mut self, _prev: Option<&Symbol<F128>>) -> Option<(OracleId, Coord)> {
        let c = *self.picked.get(self.next)?;
        self.next += 1;
        Some((ORACLE, Coord::from(c)))
    }
}

struct NoVerifier;
impl OpeningVerifier<F128> for NoVerifier {
    fn check(&self, _c: &[Claim<F128>], _a: &[(OracleId, Coord, Symbol<F128>)]) -> bool {
        true
    }
}

impl CommitmentBackend for Toy {
    type Field = F128;
    type Commitment = usize;
    type Verifier = NoVerifier;

    fn commit(&mut self, z: &[F128], ch: &mut dyn RawChallenger) -> usize {
        self.values = z.to_vec();
        self.oracles = vec![LongOracle::new(
            ORACLE,
            Alphabet::Field,
            z.iter().map(|v| Symbol(vec![*v])).collect(),
        )];
        ch.post_long_oracle(&self.oracles[0]);
        if self.mutation == Mutation::OodDoneProperly {
            let _ = ch.sample();
            let _ = ch.sample();
            ch.record_ordinary(b"ood response");
        }
        if self.mutation == Mutation::OodCoinsButNoResponse {
            let _ = ch.sample();
            let _ = ch.sample();
        }
        if self.mutation == Mutation::OodTooFewSamples {
            let _ = ch.sample();
            ch.record_ordinary(b"ood response");
        }
        if self.mutation == Mutation::UnannouncedOracle {
            self.oracles.push(LongOracle::new(
                GHOST,
                Alphabet::Field,
                vec![Symbol(vec![F128::ZERO]); 4],
            ));
        }
        if self.mutation == Mutation::ContentlessAnnouncement {
            ch.post_oracle(GHOST);
        }
        z.len()
    }

    fn open(&mut self, _claims: &[Claim<F128>], ch: &mut dyn RawChallenger) {
        if self.mutation == Mutation::OodAfterTheCut {
            let _ = ch.sample();
            ch.record_ordinary(b"ood response");
        }
        ch.record_ordinary(b"opening");
    }

    fn query_plan(&self, ch: &mut dyn RawChallenger) -> Box<dyn QueryPlan<F128>> {
        let n = self.values.len();
        match self.mutation {
            Mutation::UnsampledPlan => Box::new(Picked {
                picked: (0..self.q).map(|k| k * (n / self.q)).collect(),
                next: 0,
            }),
            Mutation::SpendsCoinsButIgnoresThem => {
                let _ = draw(ch, n, self.q);
                Box::new(Picked {
                    picked: (0..self.q).map(|k| k * (n / self.q)).collect(),
                    next: 0,
                })
            }
            Mutation::RepeatsOneSample => {
                let draws = draw(ch, n, self.q);
                Box::new(Picked {
                    picked: vec![draws[0]; self.q],
                    next: 0,
                })
            }
            Mutation::LowerHalfBias => Box::new(Picked {
                picked: draw(ch, n / 2, self.q),
                next: 0,
            }),
            Mutation::FakeFullRead => Box::new(Picked {
                picked: vec![0; n],
                next: 0,
            }),
            Mutation::PostOracleWhileSampling => {
                ch.post_oracle(GHOST);
                Box::new(Picked {
                    picked: draw(ch, n, self.q),
                    next: 0,
                })
            }
            Mutation::SpeakWhileSampling => {
                ch.record_ordinary(b"late");
                Box::new(Picked {
                    picked: draw(ch, n, self.q),
                    next: 0,
                })
            }
            Mutation::PlanTooShort => Box::new(Picked {
                picked: draw(ch, n, self.q - 1),
                next: 0,
            }),
            Mutation::PlanOutOfRange => {
                let mut p = draw(ch, n, self.q);
                p[0] = n;
                Box::new(Picked { picked: p, next: 0 })
            }
            _ => Box::new(Picked {
                picked: draw(ch, n, self.q),
                next: 0,
            }),
        }
    }

    fn oracles(&self) -> &[LongOracle<F128>] {
        &self.oracles
    }

    fn reconstruct(&self, id: OracleId, _h: &TranscriptPrefix, coord: Coord) -> Symbol<F128> {
        if self.mutation == Mutation::LyingReconstruct {
            return Symbol(vec![F128::ONE.add(
                self.values[usize::try_from(coord).expect("toy coordinate fits usize")],
            )]);
        }
        if id == GHOST {
            return Symbol(vec![F128::ZERO]);
        }
        Symbol(vec![
            self.values[usize::try_from(coord).expect("toy coordinate fits usize")],
        ])
    }

    fn verifier(&self) -> NoVerifier {
        NoVerifier
    }

    fn public_params(&self, trace_len: usize, claims: usize) -> BackendParams {
        let mut params = self.params(claims);
        let ghost = (self.mutation == Mutation::UnannouncedOracle) as u64;
        params.l_long = Coord::from(trace_len + 4 * ghost as usize);
        if self.mutation == Mutation::FakeFullRead {
            params.q_long = trace_len;
            params.a_long = trace_len as u64;
        }
        params
    }

    fn params(&self, claims: usize) -> BackendParams {
        let ghost = (self.mutation == Mutation::UnannouncedOracle) as u64;
        let q = if self.mutation == Mutation::FakeFullRead {
            self.values.len()
        } else if self.mutation == Mutation::ParamsChangeAfterCommit && !self.values.is_empty() {
            self.q + 1
        } else {
            self.q
        };
        BackendParams {
            name: "toy",
            q_long: q,
            eps_bind: 0.01,
            eps_open: if self.mutation == Mutation::EpsOpenDecreasing {
                1.0 / (claims as f64 + 1.0)
            } else {
                claims as f64 * 1e-6
            },
            regime: match self.mutation {
                Mutation::OodDoneProperly => Regime::JohnsonOod { ood_samples: 2 },
                Mutation::OodAfterTheCut => Regime::JohnsonOod { ood_samples: 1 },
                Mutation::OodTooFewSamples => Regime::JohnsonOod { ood_samples: 2 },
                Mutation::OodCoinsButNoResponse => Regime::JohnsonOod { ood_samples: 2 },
                _ => Regime::UniqueDecoding,
            },
            boolean_binding: if self.mutation == Mutation::NeedsBooleanityRows {
                BooleanBinding::RequiresExplicitRows
            } else {
                BooleanBinding::ProvedByRelation
            },
            schedule: Vec::new(),
            l_long: Coord::from(self.values.len() + 4 * ghost as usize),
            a_long: if self.mutation == Mutation::WrongALong {
                q as u64 + 1
            } else {
                q as u64
            },
            alphabet: vec![Alphabet::Field; 1 + ghost as usize],
            ordinary_bytes: 7,
            r_open: 1,
            audit: AuditMode::OneOf,
        }
    }
}

fn draw(ch: &mut dyn RawChallenger, n: usize, q: usize) -> Vec<usize> {
    (0..q)
        .map(|_| (ch.sample().to_u128() as usize) & (n - 1))
        .collect()
}

fn z() -> Vec<F128> {
    (0..64).map(|i| F128::from_bool(i % 3 == 0)).collect()
}

fn claims(z: &[F128]) -> Vec<Claim<F128>> {
    let p: Vec<F128> = (0..6).map(|i| F128::new(i as u128 * 977 + 5)).collect();
    vec![Claim {
        value: rmip_iop::mle::eval_mle(z, &p),
        point: p,
    }]
}

fn items(m: Mutation) -> Vec<u8> {
    let z = z();
    let mut b = Toy::new(m);
    let mut v: Vec<u8> = rmip_backend_conformance::check(&mut b, &z, &claims(&z), false)
        .iter()
        .map(|x| x.item)
        .collect();
    v.sort_unstable();
    v.dedup();
    v
}

#[test]
fn the_unmutated_toy_conforms() {
    assert_eq!(items(Mutation::None), Vec::<u8>::new());
}

#[test]
fn a_conforming_johnson_ood_backend_is_accepted() {
    assert_eq!(items(Mutation::OodDoneProperly), Vec::<u8>::new());
}

#[test]
fn every_mutation_is_detected_at_the_right_item() {
    let cases: &[(Mutation, &[u8])] = &[
        (Mutation::PlanTooShort, &[1, 7]),
        (Mutation::PlanOutOfRange, &[1, 7]),
        (Mutation::LyingReconstruct, &[2, 6]),
        (Mutation::UnannouncedOracle, &[3]),
        (Mutation::ContentlessAnnouncement, &[3]),
        (Mutation::EpsOpenDecreasing, &[4]),
        (Mutation::ParamsChangeAfterCommit, &[4]),
        (Mutation::OodAfterTheCut, &[4]),
        (Mutation::OodTooFewSamples, &[4]),
        (Mutation::OodCoinsButNoResponse, &[4]),
        (Mutation::NeedsBooleanityRows, &[4]),
        (Mutation::WrongALong, &[7]),
        (Mutation::PostOracleWhileSampling, &[3, 8]),
        (Mutation::SpeakWhileSampling, &[8]),
        (Mutation::UnsampledPlan, &[9]),
        (Mutation::SpendsCoinsButIgnoresThem, &[9]),
        (Mutation::RepeatsOneSample, &[9]),
        (Mutation::LowerHalfBias, &[9]),
        (Mutation::FakeFullRead, &[1]),
    ];
    for (m, expected) in cases {
        assert_eq!(&items(*m)[..], *expected, "{m:?}");
    }
}

#[test]
fn item_nine_does_not_fire_on_a_backend_that_reads_everything() {
    let z = z();
    let mut b = Toy::new(Mutation::UnsampledPlan);
    b.q = z.len();
    let v = rmip_backend_conformance::check(&mut b, &z, &claims(&z), false);
    assert!(
        !v.iter().any(|x| x.item == 9),
        "item 9 fired on a full read: {v:#?}"
    );
}

#[test]
fn nondeterminism_is_detected_at_item_five() {
    use std::sync::atomic::{AtomicU64, Ordering};
    static RUNS: AtomicU64 = AtomicU64::new(0);

    struct Drifting(Toy);
    impl CommitmentBackend for Drifting {
        type Field = F128;
        type Commitment = usize;
        type Verifier = NoVerifier;
        fn commit(&mut self, z: &[F128], ch: &mut dyn RawChallenger) -> usize {
            let drift = RUNS.fetch_add(1, Ordering::SeqCst);
            let shifted: Vec<F128> = z.iter().map(|v| v.add(F128::new(drift as u128))).collect();
            self.0.commit(&shifted, ch)
        }
        fn open(&mut self, c: &[Claim<F128>], ch: &mut dyn RawChallenger) {
            self.0.open(c, ch)
        }
        fn query_plan(&self, ch: &mut dyn RawChallenger) -> Box<dyn QueryPlan<F128>> {
            self.0.query_plan(ch)
        }
        fn oracles(&self) -> &[LongOracle<F128>] {
            self.0.oracles()
        }
        fn reconstruct(&self, i: OracleId, h: &TranscriptPrefix, c: Coord) -> Symbol<F128> {
            self.0.reconstruct(i, h, c)
        }
        fn verifier(&self) -> NoVerifier {
            NoVerifier
        }
        fn public_params(&self, trace_len: usize, claims: usize) -> BackendParams {
            self.0.public_params(trace_len, claims)
        }
        fn params(&self, k: usize) -> BackendParams {
            self.0.params(k)
        }
    }

    let z = z();
    let v = rmip_backend_conformance::check_deterministic(
        || Drifting(Toy::new(Mutation::None)),
        &z,
        &claims(&z),
    );
    assert_eq!(
        v.iter().map(|x| x.item).collect::<Vec<_>>(),
        vec![5],
        "{v:#?}"
    );

    let v =
        rmip_backend_conformance::check_deterministic(|| Toy::new(Mutation::None), &z, &claims(&z));
    assert!(v.is_empty(), "{v:#?}");
}
