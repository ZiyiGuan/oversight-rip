mod f128;
#[rustfmt::skip]
pub mod ghash;
pub mod gf64;
pub mod reference;

pub use f128::{F256Unreduced, F128, REDUCTION};
pub use gf64::F64;
pub use opening::OpeningField;

pub mod opening;

pub trait BinaryField:
    Copy + Clone + std::fmt::Debug + PartialEq + Eq + std::hash::Hash + Send + Sync + 'static
{
    const ZERO: Self;
    const ONE: Self;
    const KAPPA: usize;

    fn add(self, rhs: Self) -> Self;
    fn mul(self, rhs: Self) -> Self;
    fn square(self) -> Self;
    fn inv(self) -> Option<Self>;
    fn pow(self, exp: u128) -> Self;

    fn from_bool(b: bool) -> Self {
        if b {
            Self::ONE
        } else {
            Self::ZERO
        }
    }

    fn to_be_bytes(self) -> [u8; 16];
    fn from_be_bytes(b: [u8; 16]) -> Self;
}
