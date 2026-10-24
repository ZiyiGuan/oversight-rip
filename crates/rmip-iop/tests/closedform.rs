use rmip_field::F128;
use rmip_frontend::mnist::MnistInstance;
use rmip_frontend::r1cs::R1csConfig;
use rmip_frontend::{deoracle, layout, r1cs};
use rmip_iop::closedform::ChainMatEval;
use rmip_iop::matmle;

fn point(len: usize, seed: u128) -> Vec<F128> {
    (0..len)
        .map(|i| {
            F128::new(
                seed.wrapping_mul(0x9E37_79B9_7F4A_7C15)
                    .wrapping_add(i as u128 * 0x1234_5678_9ABC_DEF1),
            )
        })
        .collect()
}

#[test]
fn the_closed_form_is_the_scan() {
    for r in [4usize, 8, 16, 32, 64, 256, 1024, 3, 5, 7, 13, 100, 257] {
        let inst = MnistInstance::new(r);
        let (circuit, shape) = inst.build_with_shape();
        let (dc, _sites) = deoracle::deoraclize(&circuit);
        let lay = layout::TraceLayout::new(&dc);
        let sys = r1cs::emit(&dc, &lay, &R1csConfig::default());
        let cf = ChainMatEval::new(&circuit, &dc, &lay, &shape, sys.mu);
        for seed in 0..3u128 {
            let r_row = point(sys.mu, 11 + seed);
            let r_col = point(lay.nu, 97 + seed);
            let want = matmle::eval_all(&sys.a, &sys.b, &sys.c, &r_row, &r_col);
            let got = cf.eval(&r_row, &r_col);
            assert_eq!((got[0], got[1], got[2]), want, "r = {r}, seed {seed}");
        }
        let (families, explicit) = cf.sizes();
        assert!(
            families <= 11 * shape.log_inputs,
            "r = {r}: {families} families for {} levels",
            shape.log_inputs
        );
        if r.is_power_of_two() {
            assert!(
                explicit
                    <= 8 * shape.log_inputs
                        + 4 * (dc.gates.len() - dc.n - dc.ell - shape.gates())
                        + 3,
                "r = {r}: {explicit} explicit entries"
            );
        }
    }
}

#[test]
fn the_regular_popcount_keeps_the_trace_at_four_views() {
    for (r, nu) in [(256usize, 10usize), (1024, 12), (4096, 14)] {
        let inst = MnistInstance::new(r);
        let (dc, _) = deoracle::deoraclize(&inst.build());
        let lay = layout::TraceLayout::new(&dc);
        assert_eq!(lay.nu, nu, "r = {r}");
        assert_eq!(lay.n_trace, 4 * r);
    }
}
