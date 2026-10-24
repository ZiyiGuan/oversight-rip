#[test]
fn the_child_index_matches_the_scan() {
    use rmip_backend_ladder::{LadderShape, Reply};
    for nu in [8usize, 12, 17, 21] {
        let mut shape = LadderShape::for_trace(nu, 23.0);
        shape.pins = 3;
        let s = shape.schedule();
        for t in &s.tables {
            let replies = (0..shape.pins)
                .map(Reply::Pin)
                .chain([Reply::Prox, Reply::Open]);
            for reply in replies {
                assert_eq!(
                    s.child_rows(t.id, reply),
                    s.child_rows_by_scan(t.id, reply),
                    "table {} {reply:?} at nu = {nu}",
                    t.id
                );
            }
        }
    }
}

#[test]
fn the_depth_parameter_sets_the_schedule_depth() {
    use rmip_backend_ladder::LadderShape;
    let nu = 21;
    let zero = LadderShape::for_trace_at_depth(nu, 23.0, 0);
    assert_eq!(zero.depth(), 0);
    assert_eq!(zero.schedule().tables.len(), 1);
    assert_eq!((zero.log_m0, zero.log_n0()), (10, 11));
    let two = LadderShape::for_trace_at_depth(nu, 23.0, 2);
    assert_eq!(two.depth(), 2);
    assert_eq!(two.schedule().tables.len(), 1 + 2 + 2 + 1);
    assert_eq!(two.log_m0, LadderShape::for_trace(nu, 23.0).log_m0);
    let full = LadderShape::for_trace(nu, 23.0);
    assert_eq!(
        LadderShape::for_trace_at_depth(nu, 23.0, full.depth())
            .schedule()
            .tables
            .len(),
        full.schedule().tables.len()
    );
}
