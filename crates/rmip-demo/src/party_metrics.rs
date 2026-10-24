use rmip_iop::metrics::{self, AllocationOwner};
use rmip_iop::transcript::{OracleId, RawChallenger, Transcript};
use std::alloc::{GlobalAlloc, Layout, System};
use std::ptr;

pub struct MeasuredChallenger<C> {
    inner: C,
}

impl<C> MeasuredChallenger<C> {
    pub fn new(inner: C) -> Self {
        Self { inner }
    }

    pub fn into_inner(self) -> C {
        self.inner
    }
}

impl<C: RawChallenger> RawChallenger for MeasuredChallenger<C> {
    fn sample(&mut self) -> rmip_field::F128 {
        let _role = metrics::enter(metrics::Role::Verifier);
        self.inner.sample()
    }
    fn sample_many(&mut self, n: usize) -> Vec<rmip_field::F128> {
        let _role = metrics::enter(metrics::Role::Verifier);
        self.inner.sample_many(n)
    }

    fn record_ordinary(&mut self, bytes: &[u8]) {
        self.inner.record_ordinary(bytes);
    }

    fn post_oracle(&mut self, id: OracleId) {
        self.inner.post_oracle(id);
    }

    fn post_long_oracle(&mut self, oracle: &rmip_iop::backend::LongOracle<rmip_field::F128>) {
        self.inner.post_long_oracle(oracle);
    }

    fn transcript(&self) -> &Transcript {
        self.inner.transcript()
    }
}

struct Header {
    owner: Option<AllocationOwner>,
}

fn extended_layout(layout: Layout) -> Option<(Layout, usize)> {
    let alignment = layout.align().max(std::mem::align_of::<Header>());
    let prefix = std::mem::size_of::<Header>().checked_add(alignment - 1)? & !(alignment - 1);
    let extended = Layout::from_size_align(prefix.checked_add(layout.size())?, alignment).ok()?;
    Some((extended, prefix))
}

pub struct TrackingAllocator;

impl TrackingAllocator {
    unsafe fn allocate(&self, layout: Layout, zeroed: bool) -> *mut u8 {
        let Some((extended, prefix)) = extended_layout(layout) else {
            return ptr::null_mut();
        };
        let base = unsafe {
            if zeroed {
                System.alloc_zeroed(extended)
            } else {
                System.alloc(extended)
            }
        };
        if base.is_null() {
            return base;
        }
        let owner = metrics::allocation_owner();
        if let Some(owner) = &owner {
            owner.allocate(layout.size());
        }
        unsafe {
            base.cast::<Header>().write(Header { owner });
            base.add(prefix)
        }
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        unsafe { self.allocate(layout, false) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        unsafe { self.allocate(layout, true) }
    }

    unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
        let (extended, prefix) = extended_layout(layout).unwrap();
        let base = unsafe { pointer.sub(prefix) };
        let header = unsafe { base.cast::<Header>().read() };
        if let Some(owner) = &header.owner {
            owner.free(layout.size());
        }
        drop(header);
        unsafe { System.dealloc(base, extended) };
    }

    unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
        let Some(new_layout) = Layout::from_size_align(new_size, layout.align()).ok() else {
            return ptr::null_mut();
        };
        let Some((new_extended, _)) = extended_layout(new_layout) else {
            return ptr::null_mut();
        };
        let (extended, prefix) = extended_layout(layout).unwrap();
        let base = unsafe { pointer.sub(prefix) };
        let resized = unsafe { System.realloc(base, extended, new_extended.size()) };
        if resized.is_null() {
            return resized;
        }
        let header = unsafe { &*resized.cast::<Header>() };
        if let Some(owner) = &header.owner {
            if new_size >= layout.size() {
                owner.allocate(new_size - layout.size());
            } else {
                owner.free(layout.size() - new_size);
            }
        }
        unsafe { resized.add(prefix) }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rmip_iop::metrics::{enter, Role, Session};

    #[test]
    fn measured_challenger_preserves_posting_time_snapshots() {
        use rmip_field::F128;
        use rmip_iop::backend::{Alphabet, LongOracle, OracleRecordingChallenger, Symbol};
        use rmip_iop::transcript::SeededChallenger;

        let mut coins = SeededChallenger::new(91);
        let recording = OracleRecordingChallenger::new(&mut coins);
        let mut measured = MeasuredChallenger::new(recording);
        let oracle = LongOracle::new(7, Alphabet::Field, vec![Symbol(vec![F128::ONE])]);
        measured.post_long_oracle(&oracle);
        let recording = measured.into_inner();
        assert_eq!(recording.error(), None);
        assert_eq!(recording.frozen().get(7).unwrap().symbols, oracle.symbols);
    }

    #[test]
    fn allocation_ownership_survives_role_changes_and_reallocation() {
        let session = Session::start();
        let original = Layout::from_size_align(128, 64).unwrap();
        let pointer = {
            let _scope = enter(Role::MainProver);
            unsafe { TrackingAllocator.alloc_zeroed(original) }
        };
        assert!(!pointer.is_null());
        assert_eq!(pointer as usize % 64, 0);
        assert!(unsafe { std::slice::from_raw_parts(pointer, 128) }
            .iter()
            .all(|byte| *byte == 0));
        let pointer = {
            let _scope = enter(Role::Verifier);
            unsafe { TrackingAllocator.realloc(pointer, original, 256) }
        };
        assert!(!pointer.is_null());
        assert_eq!(session.report().main_prover.peak_heap_bytes, 256);
        assert_eq!(session.report().verifier.peak_heap_bytes, 0);
        {
            let _scope = enter(Role::CheckerProver);
            unsafe {
                TrackingAllocator.dealloc(pointer, Layout::from_size_align(256, 64).unwrap());
            }
        }
        let report = session.finish();
        assert_eq!(report.main_prover.live_heap_bytes, 0);
        assert_eq!(report.main_prover.allocated_heap_bytes, 256);
    }

    #[test]
    fn allocation_can_outlive_its_measurement_session() {
        let layout = Layout::from_size_align(32, 8).unwrap();
        let pointer = {
            let session = Session::start();
            let pointer = {
                let _scope = enter(Role::MainProver);
                unsafe { TrackingAllocator.alloc(layout) }
            };
            session.finish();
            pointer
        };
        unsafe { TrackingAllocator.dealloc(pointer, layout) };
    }

    #[test]
    fn realloc_shrink_and_overflow_keep_accounting_consistent() {
        let session = Session::start();
        let layout = Layout::from_size_align(256, 16).unwrap();
        let pointer = {
            let _scope = enter(Role::MainProver);
            unsafe { TrackingAllocator.alloc(layout) }
        };
        let failed = unsafe { TrackingAllocator.realloc(pointer, layout, usize::MAX) };
        assert!(failed.is_null());
        assert_eq!(session.report().main_prover.live_heap_bytes, 256);
        let pointer = unsafe { TrackingAllocator.realloc(pointer, layout, 64) };
        assert!(!pointer.is_null());
        assert_eq!(session.report().main_prover.live_heap_bytes, 64);
        unsafe {
            TrackingAllocator.dealloc(pointer, Layout::from_size_align(64, 16).unwrap());
        }
        let report = session.finish();
        assert_eq!(report.main_prover.peak_heap_bytes, 256);
        assert_eq!(report.main_prover.live_heap_bytes, 0);
    }
}
