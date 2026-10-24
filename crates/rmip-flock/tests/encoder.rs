use flock_core::merkle::HashKind;
use flock_core::pcs::{
    commit::{commit_encode, PcsParams},
    ligerito::LigeritoProfile,
};
use flock_field::F128;
use rmip_flock::encoder::{Coefficients, EncodeError, ShiftedEncoder, TableShape};

const LIMIT: usize = 1 << 20;

fn values(count: usize, mut seed: u64) -> Vec<F128> {
    let mut next = || {
        seed = seed.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = seed;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    (0..count)
        .map(|_| {
            let mut bytes = [0; F128::BYTES];
            for limb in bytes.as_chunks_mut::<8>().0 {
                limb.copy_from_slice(&next().to_le_bytes());
            }
            F128::from_le_bytes(bytes)
        })
        .collect()
}

fn scalar_evaluate(coefficients: &Coefficients, position: usize) -> Vec<F128> {
    let shape = coefficients.shape();
    let mut normalizers = Vec::new();
    for i in 0..shape.native_domain_log() {
        let mut root = F128::from_u128(1u128 << i);
        for &previous in &normalizers {
            root *= root + previous;
        }
        normalizers.push(root);
    }
    let mut point = F128::from_u128((position as u128) << shape.depth());
    let mut factors = Vec::new();
    for &root in &normalizers {
        factors.push(point * root.inv());
        point *= point + root;
    }
    let mut result = vec![F128::ZERO; shape.lanes()];
    for h in 0..shape.coefficient_count() {
        let mut basis = F128::ONE;
        for bit in 0..shape.coefficient_count().trailing_zeros() as usize {
            if h >> bit & 1 != 0 {
                basis *= factors[shape.depth() + bit];
            }
        }
        for (lane, value) in result.iter_mut().enumerate() {
            *value += basis * coefficients.values()[h * shape.lanes() + lane];
        }
    }
    result
}

#[test]
fn shifted_encoder_matches_independent_polynomials() {
    for d in 0..=7 {
        let encoder = ShiftedEncoder::new(d, LIMIT).unwrap();
        for depth in 0..=d {
            for rate in 0..=(d - depth).min(3) {
                for lanes in [1, 3, 4] {
                    let shape = TableShape::new(d, depth, 1 << (d - depth - rate), lanes).unwrap();
                    let coefficients = Coefficients::new(
                        shape,
                        values(shape.coefficient_words(), 100 * d as u64 + depth as u64),
                    )
                    .unwrap();
                    let encoded = encoder.encode(&coefficients).unwrap();
                    for position in 0..shape.rows() {
                        let independent = scalar_evaluate(&coefficients, position);
                        assert_eq!(
                            encoded.row(position).unwrap(),
                            independent,
                            "d={d}, depth={depth}, rate={rate}, lanes={lanes}, position={position}"
                        );
                        assert_eq!(
                            encoder.evaluate(&coefficients, position).unwrap(),
                            independent
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn native_encoder_correspondence_includes_non_power_of_two_live_lanes() {
    for (d, rate, lanes) in [(5, 1, 3usize), (7, 2, 4), (9, 1, 46)] {
        let shape = TableShape::new(d, 0, 1 << (d - rate), lanes).unwrap();
        let coefficients =
            Coefficients::new(shape, values(shape.coefficient_words(), 0x12345)).unwrap();
        let encoder = ShiftedEncoder::new(d, LIMIT).unwrap();
        let lane_log = lanes.next_power_of_two().trailing_zeros() as usize;
        let params = PcsParams {
            m: F128::LOG_BITS + d - rate + lane_log,
            log_inv_rate: rate,
            log_batch_size: lane_log,
            profile: LigeritoProfile::Fast,
            num_lanes: Some(lanes),
            merkle_hash: HashKind::Blake3,
        };
        assert_eq!(
            encoder.encode(&coefficients).unwrap().values(),
            commit_encode(coefficients.values(), &params)
        );
    }
}

#[test]
fn rows_finished_from_every_prefix_layer_equal_the_encoded_rows() {
    let mut shapes = Vec::new();
    for domain in [0usize, 1, 3, 6, 9] {
        for depth in 0..=domain.min(2) {
            for rate in 0..=(domain - depth).min(3) {
                for lanes in [1, 3, 8, 9] {
                    shapes.push(
                        TableShape::new(domain, depth, 1 << (domain - depth - rate), lanes)
                            .unwrap(),
                    );
                }
            }
        }
    }
    shapes.push(TableShape::new(14, 0, 1 << 13, 9).unwrap());
    shapes.push(TableShape::new(14, 1, 1 << 11, 5).unwrap());
    for shape in shapes {
        let coefficients = Coefficients::new(
            shape,
            values(
                shape.coefficient_words(),
                811 + shape.rows() as u64 + shape.lanes() as u64,
            ),
        )
        .unwrap();
        let encoder = ShiftedEncoder::new(shape.native_domain_log(), LIMIT).unwrap();
        let encoded = encoder.encode(&coefficients).unwrap();
        let positions: Vec<_> = if shape.rows() <= 512 {
            (0..shape.rows()).collect()
        } else {
            (0..97)
                .map(|i| (i * 2654435761usize) % shape.rows())
                .chain([0, shape.rows() - 1])
                .collect()
        };
        for layer in shape.log_inv_rate()..=shape.domain_log() {
            let partial = encoder.encode_prefix(&coefficients, layer).unwrap();
            assert_eq!(partial.layer(), layer);
            if layer == shape.log_inv_rate() {
                assert!(partial.values().is_empty());
            }
            if layer == shape.domain_log() && layer > shape.log_inv_rate() {
                assert_eq!(partial.values(), encoded.values(), "{shape:?}");
            }
            for &position in &positions {
                assert_eq!(
                    encoder
                        .finish_row(&coefficients, &partial, position)
                        .unwrap(),
                    encoded.row(position).unwrap(),
                    "{shape:?} layer={layer} position={position}"
                );
            }
            assert!(matches!(
                encoder.finish_row(&coefficients, &partial, shape.rows()),
                Err(EncodeError::PositionOutOfDomain { .. })
            ));
        }
        for queries in [1, 2, 37, 195, 1 << 20] {
            let layer = encoder.prefix_layer(shape, queries).unwrap();
            assert!((shape.log_inv_rate()..=shape.domain_log()).contains(&layer));
        }
        assert!(encoder
            .encode_prefix(&coefficients, shape.domain_log() + 1)
            .is_err());
        if shape.log_inv_rate() > 0 {
            assert!(encoder
                .encode_prefix(&coefficients, shape.log_inv_rate() - 1)
                .is_err());
        }
    }
    let shape = TableShape::new(10, 0, 1 << 9, 9).unwrap();
    let coefficients = Coefficients::new(shape, values(shape.coefficient_words(), 9)).unwrap();
    let capped = ShiftedEncoder::new(10, shape.encoded_words() - 1).unwrap();
    assert!(matches!(
        capped.encode_prefix(&coefficients, 5),
        Err(EncodeError::ResourceLimit { .. })
    ));
    assert_eq!(
        capped.prefix_layer(shape, 195).unwrap(),
        shape.log_inv_rate()
    );
}
