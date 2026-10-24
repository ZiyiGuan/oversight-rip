use crate::matmle;
use crate::mle::{eq_table, fold_in_place};
use crate::sumcheck::{round_evals_over, Summand};
use crate::transcript::RawChallenger;
use crate::zerocheck::RowClaims;
use rmip_field::{BinaryField, F128};
use rmip_frontend::r1cs::SparseF2;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Coefficients {
    lambda_a: F128,
    lambda_b: F128,
    lambda_c: F128,
}

impl Coefficients {
    pub fn sample(ch: &mut dyn RawChallenger) -> Self {
        Self {
            lambda_a: ch.sample(),
            lambda_b: ch.sample(),
            lambda_c: ch.sample(),
        }
    }

    pub fn lambda_a(&self) -> F128 {
        self.lambda_a
    }
    pub fn lambda_b(&self) -> F128 {
        self.lambda_b
    }
    pub fn lambda_c(&self) -> F128 {
        self.lambda_c
    }

    pub fn batch(&self, v: RowClaims) -> F128 {
        self.lambda_a
            .mul(v.v_a)
            .add(self.lambda_b.mul(v.v_b))
            .add(self.lambda_c.mul(v.v_c))
    }
}

pub struct Lincheck {
    tables: Vec<Vec<F128>>,
    vars: usize,
}

impl Lincheck {
    pub fn new(
        a: &SparseF2,
        b: &SparseF2,
        c: &SparseF2,
        r_row: &[F128],
        coeffs: Coefficients,
        z: Vec<F128>,
    ) -> Self {
        let cols = z.len();
        let eq_row = eq_table(r_row);
        let la = matmle::row_restriction(a, &eq_row, cols);
        let lb = matmle::row_restriction(b, &eq_row, cols);
        let lc = matmle::row_restriction(c, &eq_row, cols);
        let l: Vec<F128> = (0..cols)
            .map(|w| {
                coeffs
                    .lambda_a
                    .mul(la[w])
                    .add(coeffs.lambda_b.mul(lb[w]))
                    .add(coeffs.lambda_c.mul(lc[w]))
            })
            .collect();
        assert!(
            cols.is_power_of_two(),
            "column count {cols} is not 2^nu; trailing_zeros would silently give the wrong round count"
        );
        let vars = cols.trailing_zeros() as usize;
        Self {
            vars,
            tables: vec![l, z],
        }
    }

    pub fn current_sum(&self) -> F128 {
        (0..self.tables[0].len())
            .map(|w| self.tables[0][w].mul(self.tables[1][w]))
            .fold(F128::ZERO, |acc, v| acc.add(v))
    }

    pub fn z_claim(&self) -> F128 {
        self.tables[1][0]
    }

    pub fn prover_l_at_point(&self) -> F128 {
        self.tables[0][0]
    }
}

impl Summand for Lincheck {
    fn num_vars(&self) -> usize {
        self.vars
    }

    fn degree(&self) -> usize {
        2
    }

    fn round_evals(&self) -> Vec<F128> {
        round_evals_over(&self.tables, 2, |v| v[0].mul(v[1]))
    }

    fn fold(&mut self, r: F128) {
        for t in &mut self.tables {
            fold_in_place(t, r);
        }
        self.vars -= 1;
    }

    fn final_value(&self) -> F128 {
        self.tables[0][0].mul(self.tables[1][0])
    }
}

#[cfg(test)]
mod coefficient_privacy_tests {
    use super::*;
    use crate::transcript::SeededChallenger;

    const VERIFIER_MODULES: [(&str, &str); 5] = [
        ("driver.rs", include_str!("driver.rs")),
        ("relation_protocol.rs", include_str!("relation_protocol.rs")),
        ("common_protocol.rs", include_str!("common_protocol.rs")),
        ("local_protocol.rs", include_str!("local_protocol.rs")),
        ("replay.rs", include_str!("replay.rs")),
    ];

    fn code_lines(src: &str) -> impl Iterator<Item = &str> {
        src.lines().filter(|l| !l.trim_start().starts_with("//"))
    }

    fn module_code() -> &'static str {
        let src = include_str!("lincheck.rs");
        src.split("#[cfg(test)]")
            .next()
            .expect("split yields a first part")
    }

    fn braced_bodies<'a>(code: &'a str, marker: &str) -> Result<Vec<&'a str>, String> {
        let mut bodies = Vec::new();
        for (start, _) in code.match_indices(marker) {
            if marker == "Self{" && code[..start].ends_with("->") {
                continue;
            }
            let open = start + marker.len() - 1;
            let mut depth = 0usize;
            let mut close = None;
            for (offset, byte) in code.as_bytes()[open..].iter().enumerate() {
                match byte {
                    b'{' => depth += 1,
                    b'}' => {
                        depth -= 1;
                        if depth == 0 {
                            close = Some(open + offset);
                            break;
                        }
                    }
                    _ => {}
                }
            }
            let close = close.ok_or_else(|| format!("unclosed body after {marker}"))?;
            bodies.push(&code[open + 1..close]);
        }
        Ok(bodies)
    }

    fn check_coefficient_constructions(src: &str) -> Result<(), String> {
        let code: String = code_lines(src)
            .flat_map(str::chars)
            .filter(|ch| !ch.is_whitespace())
            .collect();
        let mut declarations = 0;
        let mut implementations = 0;
        for (position, _) in code.match_indices("Coefficients{") {
            let before = &code[..position];
            if before.ends_with("pubstruct") {
                declarations += 1;
            } else if before.ends_with("impl") {
                implementations += 1;
            } else {
                return Err("a struct literal names the coefficient fields directly".into());
            }
        }
        if (declarations, implementations) != (1, 1) {
            return Err(format!(
                "expected one declaration and impl, found {declarations}/{implementations}"
            ));
        }
        let implementation = braced_bodies(&code, "implCoefficients{")?[0];
        let built = braced_bodies(implementation, "Self{")?;
        if built.len() != 1 {
            return Err(format!(
                "sample must be the only constructor, found {built:?}"
            ));
        }
        let body = built[0];
        if body.matches("ch.sample()").count() != 3 {
            return Err(format!(
                "constructor must draw three separate coins: {body}"
            ));
        }
        if body.contains("mul") {
            return Err(format!("constructor multiplies a coin: {body}"));
        }
        let sample = braced_bodies(
            implementation,
            "pubfnsample(ch:&mutdynRawChallenger)->Self{",
        )?;
        if sample.len() != 1 || !sample[0].contains(&format!("Self{{{body}}}")) {
            return Err("the coefficient literal must be inside sample".into());
        }
        Ok(())
    }

    #[test]
    fn sample_is_the_only_construction_in_this_module() {
        check_coefficient_constructions(module_code()).unwrap();
    }

    #[test]
    fn construction_guard_ignores_layout_and_still_rejects_extra_literals_and_products() {
        let compact = "pub struct Coefficients {lambda_a:F128,lambda_b:F128,lambda_c:F128}
            impl Coefficients {
                pub fn sample(ch: &mut dyn RawChallenger) -> Self {
                    Self {lambda_a:ch.sample(),lambda_b:ch.sample(),lambda_c:ch.sample()}
                }
            }";
        check_coefficient_constructions(compact).unwrap();
        let multiline = compact.replace('{', "{\n").replace(',', ",\n");
        check_coefficient_constructions(&multiline).unwrap();
        let extra = compact.replace(
            "pub fn sample",
            "fn extra(ch: &mut dyn RawChallenger) -> Self {
                Self {lambda_a:ch.sample(),lambda_b:ch.sample(),lambda_c:ch.sample()}
            } pub fn sample",
        );
        assert!(check_coefficient_constructions(&extra).is_err());
        let named = compact.replace("Self {lambda_a", "Coefficients {lambda_a");
        assert!(check_coefficient_constructions(&named).is_err());
        let product = compact.replace(
            "lambda_b:ch.sample()",
            "lambda_b:ch.sample().mul(F128::ONE)",
        );
        assert!(check_coefficient_constructions(&product).is_err());
        let powers = compact.replace(
            "Self {lambda_a:ch.sample(),lambda_b:ch.sample(),lambda_c:ch.sample()}",
            "let g=ch.sample(); Self {lambda_a:g,lambda_b:g.mul(g),lambda_c:g.mul(g).mul(g)}",
        );
        assert!(check_coefficient_constructions(&powers).is_err());
    }

    #[test]
    fn the_three_coefficients_are_three_separate_coins() {
        let mut expected = SeededChallenger::new(0x1abe1);
        let (first, second, third) = (expected.sample(), expected.sample(), expected.sample());
        assert!(
            first != second && second != third && first != third,
            "degenerate seed"
        );

        let mut ch = SeededChallenger::new(0x1abe1);
        let c = Coefficients::sample(&mut ch);
        assert_eq!(c.lambda_a(), first, "lambda_a is the first coin");
        assert_eq!(c.lambda_b(), second, "lambda_b is the second coin");
        assert_eq!(c.lambda_c(), third, "lambda_c is the third coin");
        assert_eq!(
            ch.sample(),
            expected.sample(),
            "sample drew exactly three coins"
        );

        let square = c.lambda_a().mul(c.lambda_a());
        assert_ne!(
            c.lambda_b(),
            square,
            "lambda_b must not be lambda_a squared"
        );
        assert_ne!(
            c.lambda_c(),
            square.mul(c.lambda_a()),
            "lambda_c must not be lambda_a cubed"
        );
    }

    #[test]
    fn no_verifier_module_reads_the_prover_endpoint() {
        for (name, src) in VERIFIER_MODULES {
            for line in code_lines(src) {
                assert!(
                    !line.contains("prover_l_at_point"),
                    "{name} reads the prover's L(r_col) instead of recomputing it: {line:?}"
                );
            }
        }
    }
}
