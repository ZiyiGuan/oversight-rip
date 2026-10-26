use rmip_backend_interleaved::code;
use rmip_iop::transcript::OracleId;

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LadderShape {
    pub nu: usize,
    pub log_m0: usize,
    pub rate_inv: usize,
    pub lambda_bits: f64,
    pub pins: usize,
    pub log_n_min: usize,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Reply {
    Pin(usize),
    Prox,
    Open,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Half {
    Even,
    Odd,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RowSource {
    Trace(usize),
    Half {
        parent: OracleId,
        reply: Reply,
        half: Half,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kind {
    Trace,
    Pin,
    Open,
    OpenPin,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Table {
    pub id: OracleId,
    pub kind: Kind,
    pub level: usize,
    pub rows: Vec<RowSource>,
}

impl Table {
    pub fn has_open(&self) -> bool {
        matches!(self.kind, Kind::Trace | Kind::Open)
    }
}

#[derive(Clone, Debug)]
pub struct Schedule {
    pub shape: LadderShape,
    pub tables: Vec<Table>,
    pub depth: usize,
    pub queries: usize,
    child_index: ChildIndex,
}

type ChildIndex = std::collections::HashMap<(OracleId, Reply), (OracleId, usize, usize)>;
type Halves =
    std::collections::HashMap<(OracleId, Reply), (OracleId, Option<usize>, Option<usize>)>;

impl LadderShape {
    pub fn for_trace(nu: usize, lambda_bits: f64) -> Self {
        let log_m0 = (nu / 2).min(5);
        Self {
            nu,
            log_m0,
            rate_inv: 4,
            lambda_bits,
            pins: 2,
            log_n_min: 4,
        }
    }

    pub fn for_trace_at_depth(nu: usize, lambda_bits: f64, depth: usize) -> Self {
        if depth == 0 {
            let log_m0 = nu / 2;
            return Self {
                nu,
                log_m0,
                rate_inv: 4,
                lambda_bits,
                pins: 2,
                log_n_min: nu - log_m0,
            };
        }
        let mut s = Self::for_trace(nu, lambda_bits);
        s.log_n_min = s.log_n0().saturating_sub(depth);
        s
    }

    pub fn log_n0(&self) -> usize {
        assert!(
            self.log_m0 <= self.nu,
            "level 0 cannot have more rows than entries"
        );
        self.nu - self.log_m0
    }
    pub fn m0(&self) -> usize {
        1 << self.log_m0
    }
    pub fn n0(&self) -> usize {
        1 << self.log_n0()
    }
    pub fn n_at(&self, level: usize) -> usize {
        assert!(level <= self.log_n0());
        1 << (self.log_n0() - level)
    }
    pub fn n_enc_at(&self, level: usize) -> usize {
        self.n_at(level) * self.rate_inv
    }
    pub fn depth(&self) -> usize {
        self.log_n0().saturating_sub(self.log_n_min)
    }
    pub fn is_pinless(&self) -> bool {
        self.pins == 0
    }
    pub fn delta(&self) -> f64 {
        code::unique_decoding_radius(self.n0(), self.n_enc_at(0))
    }
    pub fn queries(&self) -> usize {
        code::queries_for(self.lambda_bits, self.delta())
    }

    pub fn schedule(&self) -> Schedule {
        let d = self.depth();
        assert!(
            self.pins >= 1 || d == 0,
            "a ladder of depth {d} needs at least one pinning functional; only the one-table \
             ladder (depth 0) exists without pins, under the rewinding compilation"
        );
        assert!(self.rate_inv >= 2 && self.rate_inv.is_power_of_two());
        let pins = self.pins;
        let mut tables: Vec<Table> = Vec::new();
        tables.push(Table {
            id: 0,
            kind: Kind::Trace,
            level: 0,
            rows: (0..self.m0()).map(RowSource::Trace).collect(),
        });
        let a_id = |k: usize| -> OracleId { k };
        let o_id = |k: usize| -> OracleId { d + 2 * k - 1 };
        let p_id = |k: usize| -> OracleId { d + 2 * k - 2 };
        let halves = |parent: OracleId, reply: Reply| {
            [
                RowSource::Half {
                    parent,
                    reply,
                    half: Half::Even,
                },
                RowSource::Half {
                    parent,
                    reply,
                    half: Half::Odd,
                },
            ]
        };
        let pin_rows = |parent: OracleId| -> Vec<RowSource> {
            (0..pins)
                .flat_map(|i| halves(parent, Reply::Pin(i)))
                .collect()
        };
        for k in 1..=d {
            tables.push(Table {
                id: a_id(k),
                kind: Kind::Pin,
                level: k,
                rows: pin_rows(if k == 1 { 0 } else { a_id(k - 1) }),
            });
        }
        if d >= 1 {
            tables.push(Table {
                id: o_id(1),
                kind: Kind::Open,
                level: 1,
                rows: halves(0, Reply::Prox)
                    .into_iter()
                    .chain(halves(0, Reply::Open))
                    .collect(),
            });
        }
        for k in 2..=d {
            let mut p_rows = pin_rows(o_id(k - 1));
            if k >= 3 {
                p_rows.extend(pin_rows(p_id(k - 1)));
            }
            tables.push(Table {
                id: p_id(k),
                kind: Kind::OpenPin,
                level: k,
                rows: p_rows,
            });
            let mut o_rows: Vec<RowSource> = halves(o_id(k - 1), Reply::Prox).to_vec();
            if k >= 3 {
                o_rows.extend(halves(p_id(k - 1), Reply::Prox));
            }
            o_rows.extend(halves(o_id(k - 1), Reply::Open));
            o_rows.extend(halves(a_id(k - 1), Reply::Prox));
            tables.push(Table {
                id: o_id(k),
                kind: Kind::Open,
                level: k,
                rows: o_rows,
            });
        }
        for (i, t) in tables.iter().enumerate() {
            assert_eq!(t.id, i, "ids are positions in posting order");
        }
        let child_index = child_index(&tables);
        Schedule {
            shape: *self,
            tables,
            depth: d,
            queries: self.queries(),
            child_index,
        }
    }
}

impl Schedule {
    pub fn table(&self, id: OracleId) -> &Table {
        &self.tables[id]
    }
    pub fn oracle_count(&self) -> usize {
        self.tables.len()
    }
    pub fn child_rows(&self, parent: OracleId, reply: Reply) -> Option<(OracleId, usize, usize)> {
        self.child_index.get(&(parent, reply)).copied()
    }
}

fn child_index(tables: &[Table]) -> ChildIndex {
    let mut halves: Halves = std::collections::HashMap::new();
    for t in tables {
        for (r, src) in t.rows.iter().enumerate() {
            if let RowSource::Half {
                parent,
                reply,
                half,
            } = src
            {
                let slot = halves
                    .entry((*parent, *reply))
                    .or_insert((t.id, None, None));
                assert_eq!(slot.0, t.id, "a reply's halves live in one table");
                match half {
                    Half::Even => slot.1 = Some(r),
                    Half::Odd => slot.2 = Some(r),
                }
            }
        }
    }
    halves
        .into_iter()
        .filter_map(|(key, (id, e, o))| Some((key, (id, e?, o?))))
        .collect()
}

impl Schedule {
    pub fn child_rows_by_scan(
        &self,
        parent: OracleId,
        reply: Reply,
    ) -> Option<(OracleId, usize, usize)> {
        for t in &self.tables {
            let mut even = None;
            let mut odd = None;
            for (r, src) in t.rows.iter().enumerate() {
                if let RowSource::Half {
                    parent: p,
                    reply: rp,
                    half,
                } = src
                {
                    if *p == parent && *rp == reply {
                        match half {
                            Half::Even => even = Some(r),
                            Half::Odd => odd = Some(r),
                        }
                    }
                }
            }
            if let (Some(e), Some(o)) = (even, odd) {
                return Some((t.id, e, o));
            }
        }
        None
    }
    pub fn bottom(&self) -> Vec<OracleId> {
        self.tables
            .iter()
            .filter(|t| t.level == self.depth)
            .map(|t| t.id)
            .collect()
    }
    pub fn commit_tables(&self) -> Vec<OracleId> {
        self.tables
            .iter()
            .filter(|t| matches!(t.kind, Kind::Trace | Kind::Pin))
            .map(|t| t.id)
            .collect()
    }
    pub fn a(&self, k: usize) -> OracleId {
        k
    }
    pub fn o(&self, k: usize) -> OracleId {
        self.depth + 2 * k - 1
    }
    pub fn p(&self, k: usize) -> OracleId {
        self.depth + 2 * k - 2
    }
    pub fn fold_coordinate(&self, level: usize) -> usize {
        assert!(level >= 1);
        self.shape.log_m0 + self.shape.log_n0() - level
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_schedule_has_three_chains_in_posting_order() {
        let sh = LadderShape {
            nu: 8,
            log_m0: 2,
            rate_inv: 4,
            lambda_bits: 1.0,
            pins: 2,
            log_n_min: 2,
        };
        let s = sh.schedule();
        assert_eq!(s.depth, 4);
        assert_eq!(s.oracle_count(), 12);
        let kinds: Vec<Kind> = s.tables.iter().map(|t| t.kind).collect();
        assert_eq!(
            kinds,
            vec![
                Kind::Trace,
                Kind::Pin,
                Kind::Pin,
                Kind::Pin,
                Kind::Pin,
                Kind::Open,
                Kind::OpenPin,
                Kind::Open,
                Kind::OpenPin,
                Kind::Open,
                Kind::OpenPin,
                Kind::Open
            ]
        );
        assert_eq!(s.table(s.a(1)).rows.len(), 4);
        assert_eq!(s.table(s.o(1)).rows.len(), 4);
        assert_eq!(s.table(s.p(2)).rows.len(), 4);
        assert_eq!(s.table(s.o(2)).rows.len(), 6);
        assert_eq!(s.table(s.p(3)).rows.len(), 8);
        assert_eq!(s.table(s.o(3)).rows.len(), 8);
        for t in &s.tables {
            let mut replies: Vec<Reply> = (0..sh.pins).map(Reply::Pin).collect();
            replies.push(Reply::Prox);
            if t.has_open() {
                replies.push(Reply::Open);
            }
            for r in replies {
                let home = s.child_rows(t.id, r);
                if t.level == s.depth {
                    assert!(home.is_none(), "bottom replies are in the clear");
                } else {
                    let (child, e, o) = home.expect("a home for every reply");
                    assert_eq!(s.table(child).level, t.level + 1);
                    assert_eq!(o, e + 1);
                }
            }
        }
        assert_eq!(s.bottom(), vec![s.a(4), s.p(4), s.o(4)]);
    }

    #[test]
    fn depth_zero_is_the_bundled_backend() {
        let sh = LadderShape {
            nu: 6,
            log_m0: 3,
            rate_inv: 4,
            lambda_bits: 1.0,
            pins: 1,
            log_n_min: 3,
        };
        let s = sh.schedule();
        assert_eq!(s.depth, 0);
        assert_eq!(s.oracle_count(), 1);
        assert_eq!(s.bottom(), vec![0]);
    }
}
