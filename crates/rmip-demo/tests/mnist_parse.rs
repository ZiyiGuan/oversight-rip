use rmip_demo::mnist_bench::{parse_digit_row, sparse_instance, RowError};

#[test]
fn a_digit_row_rejects_every_character_but_an_ascii_digit() {
    assert_eq!(parse_digit_row("0369"), Ok(vec![0u8, 3, 6, 9]));
    assert_eq!(parse_digit_row(""), Err(RowError::Empty));

    for (row, index, ch) in [
        ("03 9", 2usize, ' '),
        ("039\r", 3, '\r'),
        ("03a9", 2, 'a'),
        ("03:9", 2, ':'),
        ("03\u{0663}9", 2, '\u{0663}'),
    ] {
        assert_eq!(
            parse_digit_row(row),
            Err(RowError::BadChar { index, ch }),
            "row {row:?} must be rejected at character {index}"
        );
    }
}

#[test]
fn the_shipped_manifest_rows_parse_strictly() {
    for r in [256usize, 1024, 4096] {
        let inst = sparse_instance(r);
        assert_eq!(
            inst.table.len(),
            r,
            "one judge prediction per registered mask"
        );
        assert!(
            inst.table.iter().all(|&y| y < 10),
            "every table entry is a label"
        );
    }
}
