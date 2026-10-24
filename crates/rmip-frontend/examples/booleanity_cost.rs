use rmip_frontend::mnist::MnistInstance;
use rmip_frontend::r1cs::{Booleanity, R1csConfig};
use rmip_frontend::{deoracle, layout, r1cs};
use std::time::Instant;

fn main() {
    let inst = MnistInstance::new(1024);
    let c = inst.build();
    let (dc, _) = deoracle::deoraclize(&c);
    let lay = layout::TraceLayout::new(&dc);

    println!("MNIST circuit, R = {} views, N = {}\n", inst.r, lay.n_trace);
    let probe = r1cs::emit(&dc, &lay, &R1csConfig::default());
    println!(
        "rows actually emitted (m0) = {}, padded M = {}, SLACK = {}",
        probe.m0,
        probe.m,
        probe.m - probe.m0
    );
    println!(
        "ell = {} answer coordinates, tail = {} unused coordinates\n",
        dc.ell,
        lay.tail_pad_range().len()
    );
    println!(
        "{:<26} {:>8} {:>8} {:>10} {:>12}",
        "option", "M", "mu", "Z", "emit (ms)"
    );

    let mut out = Vec::new();
    for (name, bl) in [
        ("(a)/(c) none  [UNSOUND]", Booleanity::FromBinding),
        ("(e) free-source rows", Booleanity::AnswerBlockRows),
        ("(b) N R1CS rows", Booleanity::ExplicitRows),
    ] {
        let t0 = Instant::now();
        let sys = r1cs::emit(
            &dc,
            &lay,
            &R1csConfig {
                booleanity: bl,
                force_square: false,
            },
        );
        let ms = t0.elapsed().as_secs_f64() * 1000.0;
        println!(
            "{:<26} {:>8} {:>8} {:>10} {:>12.1}",
            name, sys.m, sys.mu, sys.z, ms
        );
        out.push((sys.m, sys.mu, sys.z));
    }

    let (m0, mu0, z0) = out[0];
    let (me, mu_e, ze) = out[1];
    let (m1, mu1, z1) = out[2];
    let tmat_e = me + lay.n_trace + ze;
    let tmat0 = m0 + lay.n_trace + z0;
    let tmat1 = m1 + lay.n_trace + z1;
    println!("\nT_mat proxy (M + N + Z)");
    println!("  without Booleanity rows : {tmat0}");
    println!(
        "  option (b), with rows   : {tmat1}   ({:.2}x)",
        tmat1 as f64 / tmat0 as f64
    );
    println!(
        "  option (e), source rows : {tmat_e}   ({:.3}x)",
        tmat_e as f64 / tmat0 as f64
    );
    println!("  R * T_mat is the amplified matrix-MLE component.");

    let sc0 = m0 * mu0 + lay.n_trace * lay.nu;
    let sc_b = m1 * mu1 + lay.n_trace * lay.nu;
    let sc_d = sc0 + lay.n_trace * lay.nu;

    println!(
        "\n{:<26} {:>14} {:>14}",
        "", "verifier T_mat", "prover sumcheck"
    );
    println!(
        "{:<26} {:>14} {:>14}",
        "no Booleanity [UNSOUND]",
        format!("{tmat0}"),
        format!("{sc0}")
    );
    println!(
        "{:<26} {:>14} {:>14}",
        "(b) N R1CS rows",
        format!("{tmat1} ({:.2}x)", tmat1 as f64 / tmat0 as f64),
        format!("{sc_b} ({:.2}x)", sc_b as f64 / sc0 as f64)
    );
    println!(
        "{:<26} {:>14} {:>14}",
        "(d) Booleanity sumcheck",
        format!("{tmat0} (1.00x)"),
        format!("{sc_d} ({:.2}x)", sc_d as f64 / sc0 as f64)
    );
    let sc_e = me * mu_e + lay.n_trace * lay.nu;
    println!(
        "{:<26} {:>14} {:>14}",
        "(e) free-source rows",
        format!("{tmat_e} ({:.3}x)", tmat_e as f64 / tmat0 as f64),
        format!("{sc_e} ({:.2}x)", sc_e as f64 / sc0 as f64)
    );

    println!("\noption (d) detail");
    println!("  M, Z, T_mat            : UNCHANGED, so R * T_mat is unchanged");
    println!("  extra claims           : 1  (K_rel 2 -> 3)");
    println!(
        "  extra error            : 4*nu/|K| = {}/2^128 ~ 2^-{:.0}",
        4 * lay.nu,
        128.0 - (4.0 * lay.nu as f64).log2()
    );
    println!("  option (b) adds {} nonzeros to Z alone.", z1 - z0);
    println!("\n  On the PROVER the two are close: (b) roughly doubles the R1CS");
    println!("  while (d) adds one sumcheck of comparable size. The decisive");
    println!("  difference is the VERIFIER multiplier printed above, and the");
    println!("  verifier is the party multiplied by R (R * T_mat).");
}
