use std::cell::RefCell;

use rmip_frontend::bits::BitSlice;
use rmip_frontend::oracle::{pack_query, Oracle};
use rmip_packed::statement::{packed_answers, query_bits, PackedStatement};

struct Recording<'a, O: Oracle> {
    inner: &'a O,
    queries: RefCell<Vec<Vec<bool>>>,
}

impl<O: Oracle> Oracle for Recording<'_, O> {
    fn query(&self, u: &BitSlice) -> bool {
        self.queries
            .borrow_mut()
            .push((0..u.len()).map(|k| u[k]).collect());
        self.inner.query(u)
    }
}

#[test]
fn the_statement_matches_the_circuit() {
    for r in [256usize, 4096] {
        let mut predictions = vec![7u8; r - r * 6 / 100];
        predictions.extend(vec![1u8; r * 6 / 100]);
        let certified = rmip_demo::mnist_survey::certify(0, 7, predictions).expect("certify");
        let threshold = certified.certificate.threshold;
        let (case, oracle) = certified.build_case("synthetic").expect("build the case");
        let st = PackedStatement::new(&case.inst, &case.circuit, threshold as u64, true);
        let x = case.inst.x.as_slice();
        let recording = Recording {
            inner: &oracle,
            queries: RefCell::new(Vec::new()),
        };
        let (accepted, interpreter_answers) = case.circuit.eval_with_answers(x, &recording);
        let recorded = recording.queries.into_inner();
        assert_eq!(recorded.len(), r);
        for j in [0usize, 1, 63, 64, r / 2, r - 1] {
            let q = query_bits(&st, x, j);
            assert_eq!(
                (0..q.len()).map(|k| q.get(k)).collect::<Vec<_>>(),
                recorded[j],
                "query {j}"
            );
        }
        let answers = packed_answers(&st, x, &oracle);
        assert_eq!(answers.len(), r);
        for (j, &a) in answers.iter().enumerate() {
            assert_eq!(a, interpreter_answers.get(j), "answer {j}");
        }
        let planned = PackedStatement::new(&case.inst, &case.circuit, threshold as u64, true)
            .with_query_plan();
        assert_eq!(planned.plan.as_ref().unwrap().width, 64);
        assert_eq!(packed_answers(&planned, x, &oracle), answers);
        for (j, q) in recorded.iter().enumerate().step_by(97) {
            let word = pack_query(q);
            assert_eq!(oracle.query_word(word, 64), oracle.query(q), "gate {j}");
            assert_eq!(
                oracle.query_word(word ^ 1, 64),
                oracle.query(&{
                    let mut f = q.clone();
                    f[0] = !f[0];
                    f
                })
            );
        }
        assert!(!oracle.query_word(0, 64));
        assert!(!oracle.query_word(pack_query(&recorded[0]), 63));
        let count = answers.iter().filter(|&&b| b).count() as u64;
        assert_eq!(count >= st.threshold && st.valid, accepted);
        assert!(accepted);
    }
}

#[test]
#[ignore = "measurement"]
fn witness_throughput() {
    use rmip_field::{F128, F64};
    use rmip_packed::statement::packed_answer_words;
    use rmip_packed::PackedField;
    use std::time::Instant;
    let sizes: Vec<usize> = std::env::var("RMIP_VIEWS_LOG")
        .ok()
        .map(|v| v.split(',').map(|x| x.trim().parse().unwrap()).collect())
        .unwrap_or_else(|| vec![20, 22]);
    fn levels<F: PackedField>(tree: &rmip_packed::tree::Tree, words: &[u64]) -> (f64, f64, f64) {
        let t = Instant::now();
        let w = tree.witness_from_words::<F>(words);
        let total = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let sliced = tree.slice_answer_words(words);
        let _ = tree.level_witness::<F>(0, &sliced);
        let level0 = t.elapsed().as_secs_f64();
        let t = Instant::now();
        let _ = F::to_words(&rmip_packed::majority::combined_words(tree, &w));
        (total, level0, t.elapsed().as_secs_f64())
    }
    for views_log in sizes {
        let r = 1usize << views_log;
        let mut predictions = vec![7u8; r - r * 6 / 100];
        predictions.extend(vec![1u8; r * 6 / 100]);
        let certified = rmip_demo::mnist_survey::certify(0, 7, predictions).expect("certify");
        let threshold = certified.certificate.threshold;
        let (case, oracle) = certified.build_case("synthetic").expect("build the case");
        let st = PackedStatement::new(&case.inst, &case.circuit, threshold as u64, true)
            .with_query_plan();
        let x = case.inst.x.as_slice();
        let mut best_calls = f64::MAX;
        let mut words = Vec::new();
        for _ in 0..3 {
            let t = Instant::now();
            words = packed_answer_words(&st, x, &oracle);
            best_calls = best_calls.min(t.elapsed().as_secs_f64());
        }
        let ns = |s: f64| s * 1e9 / r as f64;
        let (w64, l64, c64) = levels::<F64>(&st.tree, &words);
        let (w128, l128, c128) = levels::<F128>(&st.tree, &words);
        println!(
            "2^{views_log} views: oracle calls {:.1} ms ({:.2} ns/view); witness {:.1} ms ({:.2} ns/view, level 0 {:.1} ms) at 64 bits, {:.1} ms ({:.2} ns/view, level 0 {:.1} ms) at 128; words {:.1} / {:.1} ms",
            best_calls * 1e3,
            ns(best_calls),
            w64 * 1e3,
            ns(w64),
            l64 * 1e3,
            w128 * 1e3,
            ns(w128),
            l128 * 1e3,
            c64 * 1e3,
            c128 * 1e3
        );
    }
}
