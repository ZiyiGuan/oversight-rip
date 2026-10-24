use rmip_backend_ladder::{opening_error, SingleShotTarget};

#[test]
fn the_target_shape_searches_the_columns_upward() {
    let t40 = SingleShotTarget { bits: 40 };
    let target = 2f64.powi(-40);
    let d0 = t40
        .ladder_shape(24, Some(0), 64)
        .expect("depth 0 at 64 bits");
    assert_eq!((d0.queries(), d0.pins), (64, 15), "the policy's first pair");
    assert!(opening_error(d0, 64) <= target);
    let full = t40
        .ladder_shape(24, None, 64)
        .expect("the full depth at 2^24 words");
    assert_eq!(full.queries(), 65);
    assert!(opening_error(full, 64) <= target);
    println!(
        "full depth, 2^24 words, 64 bits, 40 bits: {} columns, {} pins, opening error 2^{:.2}",
        full.queries(),
        full.pins,
        opening_error(full, 64).log2()
    );
    assert!(t40.ladder_shape(26, None, 64).is_none());
    assert!(t40.ladder_shape(26, None, 128).is_some());
    assert!(t40.ladder_shape(26, Some(0), 64).is_some());
    let t20 = SingleShotTarget::DEFAULT;
    let d0 = t20.ladder_shape(24, Some(0), 64).unwrap();
    assert_eq!((d0.queries(), d0.pins), (34, 8));
    let full = t20.ladder_shape(24, None, 64).unwrap();
    assert_eq!((full.queries(), full.pins), (34, 12));
}
