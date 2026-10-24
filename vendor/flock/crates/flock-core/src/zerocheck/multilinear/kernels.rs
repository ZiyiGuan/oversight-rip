#[cfg(all(not(feature = "field64"), target_arch = "aarch64"))]
pub(super) mod aarch64;

#[cfg(all(feature = "field64", target_arch = "aarch64", target_feature = "aes"))]
pub(super) mod field64;

#[cfg(all(
    all(not(feature = "field64"), target_arch = "x86_64"),
    target_feature = "avx512f",
    target_feature = "vpclmulqdq"
))]
pub(super) mod x86_64;
