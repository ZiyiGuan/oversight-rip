use crate::transcript::RawChallenger;
use rmip_field::F128;
use rmip_frontend::SparseF2;

pub fn holography_pays(nu: usize) -> bool {
    nu >= 15
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct MatEvalFault;

pub trait HoloMatEval {
    #[allow(clippy::too_many_arguments)]
    fn prove_frames(
        &self,
        a: &SparseF2,
        b: &SparseF2,
        c: &SparseF2,
        r_row: &[F128],
        r_col: &[F128],
        ch: &mut dyn RawChallenger,
    ) -> ([F128; 3], Vec<Vec<F128>>);

    fn verify_frames(
        &self,
        next: &mut dyn FnMut() -> Vec<F128>,
        r_row: &[F128],
        r_col: &[F128],
        ch: &mut dyn RawChallenger,
    ) -> Result<[F128; 3], MatEvalFault>;
}
