use rmip_field::F64;
use rmip_packed::circuit::adder_block;
use rmip_packed::lincheck::BlockR1cs;
use rmip_packed::zerocheck::PackedBits;

fn lcg(state: &mut u64) -> u64 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    *state >> 11
}

fn block_instance(block: &rmip_packed::circuit::Block, z: &[bool]) -> (BlockR1cs, PackedBits<F64>) {
    let k = block.k();
    let m = block.k_log.max(7);
    let mut bits = z.to_vec();
    bits.resize(k, false);
    let all: Vec<bool> = bits.iter().copied().cycle().take(1 << m).collect();
    let r1cs = BlockR1cs::new(m, block.k_log, block.a0.clone(), block.b0.clone());
    (r1cs, PackedBits::from_bits(m, &all))
}

#[test]
fn popcount_blocks_count() {
    let mut s = 0x7_u64;
    for v in [1usize, 2, 3, 7, 16, 63, 64] {
        let block = adder_block(v, 1);
        for trial in 0..20 {
            let inputs: Vec<bool> = (0..v)
                .map(|i| match trial {
                    0 => false,
                    1 => true,
                    2 => i % 2 == 0,
                    _ => lcg(&mut s) & 1 == 1,
                })
                .collect();
            let z = block.witness(&inputs);
            let expected = inputs.iter().filter(|&&b| b).count() as u64;
            assert_eq!(block.output(&z), expected, "v = {v}");
            let (r1cs, packed) = block_instance(&block, &z);
            assert!(r1cs.satisfies(&packed), "v = {v}, trial {trial}");
        }
        println!(
            "v = {v}: k = {}, {} products, nnz {} ({} out bits)",
            block.k(),
            block.products(),
            block.nnz(),
            block.out_bits
        );
    }
    assert_eq!(
        adder_block(64, 1).k(),
        256,
        "a block of 64 views is four bits per view"
    );
}

#[test]
fn adder_blocks_add() {
    let mut s = 0x8_u64;
    for (count, bits) in [(2usize, 3usize), (4, 7), (8, 7), (64, 7)] {
        let block = adder_block(count, bits);
        for _ in 0..10 {
            let numbers: Vec<u64> = (0..count).map(|_| lcg(&mut s) % (1 << bits)).collect();
            let inputs: Vec<bool> = numbers
                .iter()
                .flat_map(|&n| (0..bits).map(move |t| (n >> t) & 1 == 1))
                .collect();
            let z = block.witness(&inputs);
            assert_eq!(block.output(&z), numbers.iter().sum::<u64>());
            let (r1cs, packed) = block_instance(&block, &z);
            assert!(r1cs.satisfies(&packed));
            for (i, &n) in numbers.iter().enumerate() {
                for t in 0..bits {
                    assert_eq!(
                        z[block.in_base + i * block.in_stride + t],
                        (n >> t) & 1 == 1
                    );
                }
            }
        }
        println!(
            "count = {count}, bits = {bits}: k = {}, out bits {}",
            block.k(),
            block.out_bits
        );
    }
}

#[test]
fn regions_are_aligned_and_reserved() {
    for (count, bits) in [(64usize, 1usize), (64, 7), (16, 10)] {
        let block = adder_block(count, bits);
        let region = block.in_stride * count.next_power_of_two();
        assert!(
            region.is_power_of_two() && block.in_base.is_multiple_of(region) && block.in_base > 0
        );
        assert!(block.in_stride.is_power_of_two());
        assert!(block.out_stride.is_power_of_two() && block.out_base == block.out_stride);
        assert!(
            block.in_base + region <= block.out_base
                || block.out_base + block.out_stride <= block.in_base
        );
        assert!(block.out_base + block.out_stride <= block.k());
        let inputs: Vec<bool> = (0..count * bits).map(|i| i % 3 == 0).collect();
        let z = block.witness(&inputs);
        let mut bad = z.clone();
        bad[block.out_base] ^= true;
        let (r1cs, packed) = block_instance(&block, &bad);
        assert!(!r1cs.satisfies(&packed), "a wrong output bit is caught");
    }
}

#[test]
fn the_program_evaluates_the_rows() {
    let mut state = 0x1234_5678_9abc_def0u64;
    let tree = rmip_packed::tree::Tree::new(1 << 18, 6, 6);
    for level in &tree.levels {
        let block = &level.block;
        let per_block = block.in_count * block.in_bits;
        assert!(
            block.program_size() < block.nnz(),
            "the program is the cheaper evaluation"
        );
        for round in 0..20 {
            let inputs: Vec<u64> = (0..per_block)
                .map(|_| match round % 4 {
                    0 => lcg(&mut state),
                    1 => 0,
                    2 => u64::MAX,
                    _ => lcg(&mut state) & lcg(&mut state),
                })
                .collect();
            let fast = block.witness_sliced(&inputs);
            let slow = block.witness_sliced_from_rows(&inputs);
            assert!(fast.z == slow.z && fast.a == slow.a && fast.b == slow.b);
            for lane in [0usize, 17, 63] {
                let bits: Vec<bool> = inputs.iter().map(|w| (w >> lane) & 1 == 1).collect();
                let z = block.witness(&bits);
                for (s, &zs) in z.iter().enumerate() {
                    assert_eq!(zs, (fast.z[s] >> lane) & 1 == 1, "lane {lane} slot {s}");
                }
            }
        }
    }
}

#[test]
fn padding_bits_are_constrained_to_zero() {
    let b = adder_block(64, 1);
    let z = b.witness(&[false; 64]);
    let (r1cs, honest) = block_instance(&b, &z);
    assert!(r1cs.satisfies(&honest));
    for s in b.out_base + b.out_bits..b.out_base + b.out_stride {
        assert!(!z[s], "the honest padding bit {s} is zero");
        let mut forged = z.clone();
        forged[s] = true;
        let (_, instance) = block_instance(&b, &forged);
        assert!(
            !r1cs.satisfies(&instance),
            "padding bit {s} forced to one must be rejected"
        );
        assert_eq!(b.output(&forged), 0);
    }
    let mut forged = z.clone();
    let gap = (1..b.in_base)
        .find(|&s| !z[s])
        .expect("a zero slot before the inputs");
    forged[gap] = true;
    let (_, instance) = block_instance(&b, &forged);
    assert!(
        !r1cs.satisfies(&instance),
        "slot {gap} forced to one must be rejected"
    );
}
