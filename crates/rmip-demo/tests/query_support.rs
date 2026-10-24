use rmip_demo::harness::verifier_query_support;
use rmip_field::F128;

fn boolean_trace() -> Vec<F128> {
    vec![F128::ZERO, F128::ZERO, F128::ONE, F128::ONE, F128::ZERO]
}

#[test]
fn repeated_query_bundles_collapse_to_one_support_element() {
    let z = boolean_trace();
    let loc_same = [vec![0, 2, 4], vec![0, 2, 3], vec![1, 3, 4]];
    assert_eq!(
        verifier_query_support(loc_same.iter().cloned(), &z),
        1,
        "three rows reading the same bits are one support element"
    );

    let loc_two = [vec![0, 2, 4], vec![0, 2, 3], vec![2, 3, 4]];
    assert_eq!(
        verifier_query_support(loc_two.iter().cloned(), &z),
        2,
        "a row reading a different bit bundle is a second support element"
    );

    let loc_all = [vec![0, 0, 4], vec![0, 2, 4], vec![2, 2, 4]];
    assert_eq!(
        verifier_query_support(loc_all.iter().cloned(), &z),
        3,
        "three distinct bundles"
    );
}

#[test]
#[should_panic(expected = "Boolean")]
fn a_non_boolean_query_coordinate_is_refused() {
    let mut z = boolean_trace();
    z[2] = F128::new(7);
    let loc = [vec![0, 2, 4], vec![0, 0, 4]];
    let _ = verifier_query_support(loc.iter().cloned(), &z);
}
