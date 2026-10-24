use std::sync::atomic::{AtomicUsize, Ordering};

use rmip_field::F128;
use rmip_iop::backend::{CommitmentBackend, Coord, Symbol};
use rmip_iop::driver::{Execution, Occurrence};
use rmip_iop::transcript::{Entry, OracleId, RawChallenger, TranscriptPrefix};

#[derive(Clone, Copy, Debug)]
pub struct SharedAdvice {
    pub seed: u64,
}

pub struct PCheck<B: CommitmentBackend> {
    backend: B,
    questions: AtomicUsize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct IllegalPrefix {
    pub oracle: OracleId,
}

impl std::fmt::Display for IllegalPrefix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "the prefix handed to the checker is not oracle {}'s posting prefix",
            self.oracle
        )
    }
}

impl std::error::Error for IllegalPrefix {}

impl<B: CommitmentBackend> PCheck<B> {
    pub fn new(backend: B) -> Self {
        Self {
            backend,
            questions: AtomicUsize::new(0),
        }
    }

    pub fn questions_asked(&self) -> usize {
        self.questions.load(Ordering::Relaxed)
    }

    pub fn backend(&self) -> &B {
        &self.backend
    }

    pub fn answer(&self, i: OracleId, h: &TranscriptPrefix, theta: Coord) -> Symbol<B::Field>
    where
        B: CommitmentBackend<Field = F128>,
    {
        match self.try_answer(i, h, theta) {
            Ok(s) => s,
            Err(e) => panic!("{e}"),
        }
    }

    pub fn try_answer(
        &self,
        i: OracleId,
        h: &TranscriptPrefix,
        theta: Coord,
    ) -> Result<Symbol<B::Field>, IllegalPrefix>
    where
        B: CommitmentBackend<Field = F128>,
    {
        let _role = rmip_iop::metrics::enter(rmip_iop::metrics::Role::CheckerProver);
        self.questions.fetch_add(1, Ordering::Relaxed);
        let entries = h.decode().map_err(|_| IllegalPrefix { oracle: i })?;
        if entries
            .iter()
            .any(|e| matches!(e, Entry::OraclePost(id) if *id == i))
        {
            return Err(IllegalPrefix { oracle: i });
        }
        if i == rmip_iop::driver::DUMMY_ORACLE {
            return Ok(rmip_iop::driver::dummy_symbol());
        }
        Ok(self.backend.reconstruct(i, h, theta))
    }
}

#[derive(Debug)]
pub enum Audit {
    Accepted { audited: usize },
    UnderlyingRejected { audited: usize },
    Inconsistent { audited: usize },
}

impl Audit {
    pub fn accepted(&self) -> bool {
        matches!(self, Audit::Accepted { .. })
    }

    pub fn consistent(&self) -> bool {
        !matches!(self, Audit::Inconsistent { .. })
    }
}

pub fn audit<B: CommitmentBackend<Field = F128>>(
    execution: &Execution,
    checker: &PCheck<B>,
    ch: &mut dyn RawChallenger,
) -> Audit {
    audit_with_answer(execution, checker, ch).0
}

pub fn audit_with_answer<B: CommitmentBackend<Field = F128>>(
    execution: &Execution,
    checker: &PCheck<B>,
    ch: &mut dyn RawChallenger,
) -> (Audit, Symbol<F128>) {
    let q = execution.q_long();
    assert!(q >= 1, "Step 3 requires q >= 1 occurrences");
    let j = rmip_iop::sampling::uniform_index(ch, q);
    let Occurrence {
        oracle,
        prefix,
        coord,
        answer,
    } = &execution.occurrences[j];
    let (oracle, coord) = (*oracle, *coord);

    let reply = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        checker.try_answer(oracle, prefix, coord)
    }));
    let b_j = match reply {
        Ok(Ok(symbol)) => symbol,
        Ok(Err(_)) | Err(_) => {
            return (Audit::Inconsistent { audited: j }, Symbol(Vec::new()));
        }
    };

    let verdict = if &b_j != answer {
        Audit::Inconsistent { audited: j }
    } else if execution.iop_accepted() {
        Audit::Accepted { audited: j }
    } else {
        Audit::UnderlyingRejected { audited: j }
    };
    (verdict, b_j)
}

pub fn audit_all_with_answers<B: CommitmentBackend<Field = F128>>(
    execution: &Execution,
    checker: &PCheck<B>,
) -> (Audit, Vec<Symbol<F128>>) {
    let q = execution.q_long();
    assert!(q >= 1, "Step 3 requires q >= 1 occurrences");
    let mut answers = Vec::with_capacity(q);
    let mut seen: Vec<OracleId> = Vec::with_capacity(q);
    let mut last_len = 0usize;
    for (k, occ) in execution.occurrences.iter().enumerate() {
        let Occurrence {
            oracle,
            prefix,
            coord,
            answer,
        } = occ;
        if seen.contains(oracle) || prefix.entry_count() < last_len {
            return (Audit::Inconsistent { audited: k }, answers);
        }
        seen.push(*oracle);
        last_len = prefix.entry_count();
        let reply = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            checker.try_answer(*oracle, prefix, *coord)
        }));
        let b_k = match reply {
            Ok(Ok(symbol)) => symbol,
            Ok(Err(_)) | Err(_) => {
                answers.push(Symbol(Vec::new()));
                return (Audit::Inconsistent { audited: k }, answers);
            }
        };
        let agrees = &b_k == answer;
        answers.push(b_k);
        if !agrees {
            return (Audit::Inconsistent { audited: k }, answers);
        }
    }
    let verdict = if execution.iop_accepted() {
        Audit::Accepted { audited: q }
    } else {
        Audit::UnderlyingRejected { audited: q }
    };
    (verdict, answers)
}

pub fn audit_all<B: CommitmentBackend<Field = F128>>(
    execution: &Execution,
    checker: &PCheck<B>,
) -> Audit {
    audit_all_with_answers(execution, checker).0
}
