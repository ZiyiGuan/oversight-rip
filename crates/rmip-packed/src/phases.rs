use std::time::Duration;

#[derive(Clone, Debug, Default)]
pub struct Phases {
    pub level: usize,
    entries: Vec<(&'static str, usize, Duration)>,
}

impl Phases {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add(&mut self, name: &'static str, d: Duration) {
        let level = self.level;
        match self
            .entries
            .iter_mut()
            .find(|(n, l, _)| *n == name && *l == level)
        {
            Some(e) => e.2 += d,
            None => self.entries.push((name, level, d)),
        }
    }

    pub fn entries(&self) -> &[(&'static str, usize, Duration)] {
        &self.entries
    }

    pub fn total(&self, name: &str) -> Duration {
        self.entries
            .iter()
            .filter(|(n, _, _)| *n == name)
            .map(|(_, _, d)| *d)
            .sum()
    }

    pub fn report(&self) -> String {
        let mut lines = Vec::new();
        let mut upper: Vec<(&'static str, Duration)> = Vec::new();
        for &(name, level, d) in &self.entries {
            if level == 0 {
                lines.push(format!("{name}: {:.2} ms", d.as_secs_f64() * 1e3));
            } else {
                match upper.iter_mut().find(|(n, _)| *n == name) {
                    Some(e) => e.1 += d,
                    None => upper.push((name, d)),
                }
            }
        }
        for (name, d) in upper {
            lines.push(format!(
                "{name} (levels above 0): {:.2} ms",
                d.as_secs_f64() * 1e3
            ));
        }
        lines.join("; ")
    }
}
