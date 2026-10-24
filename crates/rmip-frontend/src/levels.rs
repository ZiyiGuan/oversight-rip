use crate::circuit::OracleCircuit;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LevelDecomposition {
    levels: Vec<Vec<usize>>,
}

impl LevelDecomposition {
    pub fn depth(&self) -> usize {
        self.levels.len()
    }
}

pub fn decompose(c: &OracleCircuit) -> LevelDecomposition {
    use crate::circuit::Gate;

    let mut depth_out = vec![0usize; c.gates.len()];
    let mut level = vec![0usize; c.ell()];

    for (i, g) in c.gates.iter().enumerate() {
        let ops: &[usize] = match g {
            Gate::Input(_) | Gate::Const(_) | Gate::Answer(_) => &[],
            Gate::Not(p) => std::slice::from_ref(p),
            Gate::And(p, _) | Gate::Or(p, _) | Gate::Xor(p, _) => std::slice::from_ref(p),
            Gate::Xor3(p, _, _) | Gate::Maj3(p, _, _) => std::slice::from_ref(p),
            Gate::Oracle { query, .. } => query,
        };
        let mut depth_in = ops.iter().map(|&p| depth_out[p]).max().unwrap_or(0);
        if let Gate::And(_, q) | Gate::Or(_, q) | Gate::Xor(_, q) = g {
            depth_in = depth_in.max(depth_out[*q]);
        }
        if let Gate::Xor3(_, q, s) | Gate::Maj3(_, q, s) = g {
            depth_in = depth_in.max(depth_out[*q]).max(depth_out[*s]);
        }
        depth_out[i] = match g {
            Gate::Oracle { occurrence, .. } => {
                level[*occurrence] = depth_in;
                depth_in + 1
            }
            _ => depth_in,
        };
    }

    let d = level.iter().copied().max().map(|m| m + 1).unwrap_or(0);
    let mut levels = vec![Vec::new(); d];
    for (j, &l) in level.iter().enumerate() {
        levels[l].push(j);
    }
    LevelDecomposition { levels }
}
