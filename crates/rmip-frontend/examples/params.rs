use rmip_frontend::mnist::MnistInstance;
use rmip_frontend::r1cs::{Booleanity, R1csConfig};
use rmip_frontend::{deoracle, layout, levels, loc, r1cs, report};

const SIGMA_LOC: f64 = 0.125;

fn tag(bl: Booleanity) -> &'static str {
    match bl {
        Booleanity::FromBinding => "binding",
        Booleanity::ClosedForm => "closed-form",
        Booleanity::AnswerBlockRows => "source rows",
        Booleanity::ExplicitRows => "+bool rows",
    }
}

fn row(
    name: &str,
    c: &rmip_frontend::circuit::OracleCircuit,
    eps: rmip_frontend::Ratio,
    sigma_loc: f64,
) {
    let (dc, sites) = deoracle::deoraclize(c);
    let lay = layout::TraceLayout::new(&dc);
    let l = loc::build(&sites, &lay);
    let t = report::local_sampling_repetitions(eps, sigma_loc);
    for bl in [R1csConfig::default().booleanity, Booleanity::ExplicitRows] {
        let sys = r1cs::emit(
            &dc,
            &lay,
            &R1csConfig {
                booleanity: bl,
                force_square: false,
            },
        );
        println!(
            "{:<20} {:<11} ell={:<5} b_max={:<3} k={} nu={:<3} N={:<6} mu={:<3} M={:<6} Z={:<7} g={:<6} d={} B_tr={:<7} eps={}/{} t={} K_max={}",
            name, tag(bl), dc.ell, c.b_max(), lay.k, lay.nu, lay.n_trace, sys.mu, sys.m, sys.z,
            lay.g, levels::decompose(c).depth(), loc::b_tr(&l), eps.num, eps.den, t,
            t * (c.b_max() + 1)
        );
    }
}

fn main() {
    for r in [256usize, 1024, 4096] {
        let inst = MnistInstance::new(r);
        row(
            &format!("MNIST R={r}"),
            &inst.build(),
            inst.eps_rob,
            SIGMA_LOC,
        );
    }
}
