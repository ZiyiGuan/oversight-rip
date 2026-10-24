use rmip_backend_ladder::{Ladder, LadderShape};
use rmip_iop::backend::CommitmentBackend;
use rmip_iop::driver::{Cheat, Config};
use rmip_iop::transcript::OsChallenger;
use std::time::Instant;

fn main() {
    let views: Vec<usize> = std::env::args()
        .skip(1)
        .map(|a| a.parse().expect("view count"))
        .collect();
    let views = if views.is_empty() {
        vec![4096, 16384]
    } else {
        views
    };
    for r in views {
        let t0 = Instant::now();
        let (c, oracle) = if r <= rmip_frontend::mnist::MAX_COMPILED_VIEWS {
            rmip_demo::harness::mnist_case(r)
        } else {
            let mut predictions = vec![7u8; r - r * 6 / 100];
            predictions.extend(vec![1u8; r * 6 / 100]);
            let certified = rmip_demo::mnist_survey::certify(0, 7, predictions).expect("certify");
            certified.build_case("synthetic").expect("case")
        };
        let setup = t0.elapsed();
        let st = rmip_demo::harness::statement(&c);
        let cfg = Config {
            t: c.t,
            ..Config::default()
        };
        let nu = c.layout.nu;
        println!("\n=== {r} views, nu = {nu}, case setup {setup:?} ===");
        {
            let shape = LadderShape::for_trace(nu, 4.0);
            let sched = shape.schedule();
            let checker = {
                let mut b = Ladder::lazy(shape);
                let t = Instant::now();
                b.commit(&c.honest_trace, &mut OsChallenger::new());
                println!("ladder    checker commit {:?}", t.elapsed());
                rmip_mip::PCheck::new(b)
            };
            let mut backend = Ladder::lazy(shape);
            let t = Instant::now();
            let (exec, verdict) = rmip_mip_prover::execute_recording(
                &st,
                &mut backend,
                &oracle,
                &cfg,
                &Cheat::default(),
                &mut OsChallenger::new(),
            );
            let joint = t.elapsed();
            assert!(verdict.is_ok(), "{verdict:?}");
            let t = Instant::now();
            let audit = rmip_mip::audit_all(&exec, &checker);
            let audit_t = t.elapsed();
            assert!(audit.accepted(), "{audit:?}");
            let reads: usize = exec.occurrences.iter().map(|o| o.answer.0.len()).sum();
            let clear: usize = sched.shape.n_at(sched.depth) * (3 * sched.shape.pins + 4);
            let p = backend.params(1);
            let c_pref: usize = exec
                .occurrences
                .iter()
                .map(|o| o.prefix.len())
                .max()
                .unwrap_or(0);
            println!(
                "ladder    joint {joint:?}  t_open {:?}  t_verify {:?}  t_mat {:?}  audit {audit_t:?}  reads {reads} + clear {clear}  q_long {} (depth {})  r_IOP {}  C_short {}  C_pref {c_pref}",
                exec.t_open, exec.t_verify, exec.t_mat, p.q_long, sched.depth, exec.rounds, exec.short_bytes
            );
        }
    }
}
