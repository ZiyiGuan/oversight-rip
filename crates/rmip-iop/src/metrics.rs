use std::cell::RefCell;
use std::marker::PhantomData;
use std::rc::Rc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

const ROLE_COUNT: usize = 5;
static HEAP_TRACKING: AtomicBool = AtomicBool::new(false);

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(usize)]
pub enum Role {
    MainProver,
    Verifier,
    CheckerProver,
    Diagnostic,
    Unattributed,
}

#[derive(Default)]
struct Counters {
    live: [AtomicUsize; ROLE_COUNT],
    peak: [AtomicUsize; ROLE_COUNT],
    allocated: [AtomicUsize; ROLE_COUNT],
    nanos: [AtomicU64; ROLE_COUNT],
}

struct Active {
    counters: Arc<Counters>,
    role: Role,
    since: Instant,
}

impl Active {
    fn record_until(&self, now: Instant) {
        self.counters.nanos[self.role as usize].fetch_add(
            now.duration_since(self.since)
                .as_nanos()
                .try_into()
                .unwrap(),
            Ordering::Relaxed,
        );
    }
}

thread_local! {
    static ACTIVE: RefCell<Option<Active>> = const { RefCell::new(None) };
}

#[derive(Clone, Copy, Debug)]
pub struct Stats {
    pub elapsed: Duration,
    pub peak_heap_bytes: usize,
    pub live_heap_bytes: usize,
    pub allocated_heap_bytes: usize,
}

#[derive(Clone, Copy, Debug)]
pub struct Report {
    pub main_prover: Stats,
    pub verifier: Stats,
    pub checker_prover: Stats,
    pub diagnostic: Stats,
    pub unattributed: Stats,
    pub heap_tracking: bool,
}

impl Report {
    pub fn for_role(&self, role: Role) -> Stats {
        match role {
            Role::MainProver => self.main_prover,
            Role::Verifier => self.verifier,
            Role::CheckerProver => self.checker_prover,
            Role::Diagnostic => self.diagnostic,
            Role::Unattributed => self.unattributed,
        }
    }
}

pub struct Session {
    counters: Arc<Counters>,
    previous: Option<Option<Active>>,
    _not_send: PhantomData<Rc<()>>,
}

impl Session {
    pub fn start() -> Self {
        let counters = Arc::new(Counters::default());
        let now = Instant::now();
        let previous = ACTIVE.with(|active| {
            let previous = active.replace(Some(Active {
                counters: Arc::clone(&counters),
                role: Role::Unattributed,
                since: now,
            }));
            if let Some(previous) = &previous {
                previous.record_until(now);
            }
            previous
        });
        Self {
            counters,
            previous: Some(previous),
            _not_send: PhantomData,
        }
    }

    pub fn report(&self) -> Report {
        let stats = |role: Role| {
            let index = role as usize;
            Stats {
                elapsed: Duration::from_nanos(self.counters.nanos[index].load(Ordering::Relaxed)),
                peak_heap_bytes: self.counters.peak[index].load(Ordering::Relaxed),
                live_heap_bytes: self.counters.live[index].load(Ordering::Relaxed),
                allocated_heap_bytes: self.counters.allocated[index].load(Ordering::Relaxed),
            }
        };
        Report {
            main_prover: stats(Role::MainProver),
            verifier: stats(Role::Verifier),
            checker_prover: stats(Role::CheckerProver),
            diagnostic: stats(Role::Diagnostic),
            unattributed: stats(Role::Unattributed),
            heap_tracking: HEAP_TRACKING.load(Ordering::Relaxed),
        }
    }

    pub fn finish(mut self) -> Report {
        self.stop();
        self.report()
    }

    fn stop(&mut self) {
        if let Some(mut previous) = self.previous.take() {
            let now = Instant::now();
            if let Some(previous) = &mut previous {
                previous.since = now;
            }
            ACTIVE.with(|active| {
                if let Some(current) = active.replace(previous) {
                    current.record_until(now);
                }
            });
        }
    }
}

impl Drop for Session {
    fn drop(&mut self) {
        self.stop();
    }
}

pub struct Guard {
    previous: Option<Active>,
    enabled: bool,
    _not_send: PhantomData<Rc<()>>,
}

pub fn elapsed_now(role: Role) -> Option<Duration> {
    ACTIVE.with(|active| {
        let active = active.borrow();
        let current = active.as_ref()?;
        let index = role as usize;
        let mut nanos = current.counters.nanos[index].load(Ordering::Relaxed);
        if current.role == role {
            nanos += Instant::now().duration_since(current.since).as_nanos() as u64;
        }
        Some(Duration::from_nanos(nanos))
    })
}

pub fn enter(role: Role) -> Guard {
    ACTIVE.with(|active| {
        let mut active = active.borrow_mut();
        let Some(current) = active.as_ref() else {
            return Guard {
                previous: None,
                enabled: false,
                _not_send: PhantomData,
            };
        };
        let now = Instant::now();
        current.record_until(now);
        let next = Active {
            counters: Arc::clone(&current.counters),
            role: if current.role == Role::Diagnostic {
                Role::Diagnostic
            } else {
                role
            },
            since: now,
        };
        Guard {
            previous: active.replace(next),
            enabled: true,
            _not_send: PhantomData,
        }
    })
}

impl Drop for Guard {
    fn drop(&mut self) {
        if self.enabled {
            let now = Instant::now();
            let mut previous = self.previous.take();
            if let Some(previous) = &mut previous {
                previous.since = now;
            }
            ACTIVE.with(|active| {
                if let Some(current) = active.replace(previous) {
                    current.record_until(now);
                }
            });
        }
    }
}

pub struct AllocationOwner {
    counters: Arc<Counters>,
    role: usize,
}

impl AllocationOwner {
    pub fn allocate(&self, bytes: usize) {
        let live = self.counters.live[self.role].fetch_add(bytes, Ordering::Relaxed) + bytes;
        self.counters.peak[self.role].fetch_max(live, Ordering::Relaxed);
        self.counters.allocated[self.role].fetch_add(bytes, Ordering::Relaxed);
    }

    pub fn free(&self, bytes: usize) {
        self.counters.live[self.role].fetch_sub(bytes, Ordering::Relaxed);
    }
}

pub fn allocation_owner() -> Option<AllocationOwner> {
    HEAP_TRACKING.store(true, Ordering::Relaxed);
    ACTIVE
        .try_with(|active| {
            let active = active.try_borrow().ok()?;
            let active = active.as_ref()?;
            Some(AllocationOwner {
                counters: Arc::clone(&active.counters),
                role: active.role as usize,
            })
        })
        .ok()
        .flatten()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn busy_interval() {
        let start = Instant::now();
        while start.elapsed() < Duration::from_millis(1) {
            std::hint::spin_loop();
        }
    }

    #[test]
    fn role_intervals_are_disjoint_and_diagnostics_stay_separate() {
        let wall = Instant::now();
        let session = Session::start();
        {
            let _verifier = enter(Role::Verifier);
            busy_interval();
            {
                let _prover = enter(Role::MainProver);
                busy_interval();
            }
            {
                let _diagnostic = enter(Role::Diagnostic);
                let _nested = enter(Role::Verifier);
                busy_interval();
            }
        }
        let report = session.finish();
        assert!(report.main_prover.elapsed >= Duration::from_millis(1));
        assert!(report.verifier.elapsed >= Duration::from_millis(1));
        assert!(report.diagnostic.elapsed >= Duration::from_millis(1));
        assert_eq!(report.checker_prover.elapsed, Duration::ZERO);
        assert!(
            report.main_prover.elapsed
                + report.verifier.elapsed
                + report.diagnostic.elapsed
                + report.unattributed.elapsed
                <= wall.elapsed()
        );
    }

    #[test]
    fn ownership_survives_role_changes_and_session_completion() {
        let session = Session::start();
        let owner = {
            let _prover = enter(Role::MainProver);
            let owner = allocation_owner().unwrap();
            owner.allocate(128);
            owner
        };
        {
            let _verifier = enter(Role::Verifier);
            owner.allocate(128);
            owner.free(64);
        }
        let report = session.finish();
        assert_eq!(report.main_prover.peak_heap_bytes, 256);
        assert_eq!(report.main_prover.live_heap_bytes, 192);
        assert_eq!(report.verifier.peak_heap_bytes, 0);
        std::thread::spawn(move || owner.free(192)).join().unwrap();
    }
}
