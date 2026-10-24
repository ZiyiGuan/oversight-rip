use rmip_iop::backend::Coord;
use rmip_iop::transcript::RawChallenger;

pub fn domain_size(n_enc: usize, q: usize) -> Option<Coord> {
    let q = u32::try_from(q).ok()?;
    Coord::checked_pow_u64(u64::try_from(n_enc).ok()?, q)
}

pub fn encode(j: &[usize], n_enc: usize) -> Coord {
    let radix = u64::try_from(n_enc).expect("n_enc fits u64");
    let mut theta = Coord::ZERO;
    for &jk in j.iter().rev() {
        assert!(jk < n_enc, "tuple digit {jk} outside a domain of {n_enc}");
        theta = theta
            .checked_mul_u64(radix)
            .and_then(|t| t.checked_add_u64(jk as u64))
            .expect("tuple coordinate overflows 1024 bits");
    }
    theta
}

pub fn try_decode(theta: Coord, n_enc: usize, q: usize) -> Option<Vec<usize>> {
    if n_enc == 0 {
        return None;
    }
    if n_enc.is_power_of_two() {
        let width = n_enc.trailing_zeros();
        let total = u32::try_from(q).ok()?.checked_mul(width)?;
        if total > 1024 || theta.bits() > total {
            return None;
        }
        return (0..q)
            .map(|k| usize::try_from(theta.bits_at(k as u32 * width, width)).ok())
            .collect();
    }
    let domain = domain_size(n_enc, q)?;
    if theta >= domain {
        return None;
    }
    let radix = u64::try_from(n_enc).ok()?;
    let mut rest = theta;
    let mut j = Vec::with_capacity(q);
    for _ in 0..q {
        let (quot, rem) = rest.div_rem_u64(radix);
        j.push(usize::try_from(rem).ok()?);
        rest = quot;
    }
    Some(j)
}

pub fn draw_tuple(ch: &mut dyn RawChallenger, n_enc: usize, q: usize) -> Vec<usize> {
    (0..q)
        .map(|_| rmip_backend_interleaved::draw_column(ch, n_enc))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_toy_tuple_round_trips() {
        let (n_enc, q) = (16usize, 3usize);
        let domain = domain_size(n_enc, q).unwrap();
        assert_eq!(domain, Coord::from(4096u64));
        for t in 0..4096u64 {
            let theta = Coord::from(t);
            let j = try_decode(theta, n_enc, q).unwrap();
            assert_eq!(encode(&j, n_enc), theta);
        }
        assert_eq!(try_decode(domain, n_enc, q), None);
        assert_eq!(try_decode(Coord::MAX, n_enc, q), None);
        assert_eq!(try_decode(Coord::ZERO, 0, q), None);
    }

    #[test]
    fn the_power_of_two_path_matches_the_general_decoder() {
        let mut s = 0x2545_f491_4f6c_dd1du64;
        for (n_enc, q) in [
            (16usize, 3usize),
            (1 << 10, 7),
            (1 << 21, 34),
            (1 << 30, 34),
        ] {
            for _ in 0..20 {
                let j: Vec<usize> = (0..q)
                    .map(|_| {
                        s ^= s << 13;
                        s ^= s >> 7;
                        s ^= s << 17;
                        (s as usize) % n_enc
                    })
                    .collect();
                let theta = encode(&j, n_enc);
                assert_eq!(try_decode(theta, n_enc, q).as_deref(), Some(j.as_slice()));
            }
            let domain = domain_size(n_enc, q).unwrap();
            assert_eq!(try_decode(domain, n_enc, q), None);
            assert_eq!(
                try_decode(domain.checked_sub_u64(1).unwrap(), n_enc, q).map(|j| j.len()),
                Some(q)
            );
        }
    }

    #[test]
    fn an_overflowing_domain_is_none() {
        assert_eq!(domain_size(1 << 31, 34), None);
        assert_eq!(try_decode(Coord::ZERO, 1 << 31, 34), None);
        assert_eq!(domain_size(1 << 30, 34), Some(Coord::pow2(1020)));
        assert_eq!(domain_size(1 << 21, 6), Some(Coord::from(1u128 << 126)));
        assert!(!domain_size(1 << 43, 6).unwrap().fits_u256());
        assert!(!domain_size(1 << 29, 6).unwrap().fits_u128());
    }
}
