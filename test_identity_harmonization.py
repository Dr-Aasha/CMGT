from src.manifest import (
    build_growth_identity_index,
    harmonize_target_id,
)

def check(target, growth_ids, expected, method=None, year=None):
    idx = build_growth_identity_index(growth_ids)
    got, how = harmonize_target_id(target, idx, year=year)
    assert got == expected, (target, expected, got, how)
    if method is not None:
        assert how == method, (target, method, how)
    print(f"PASS: {target} -> {got} [{how}]")

def main():
    # 2023 separator/numeric form
    check(
        "1-2-1",
        ["121", "122", "123"],
        "121",
        "compact_separator",
    )

    check(
        "5-2-9",
        ["527", "528", "529"],
        "529",
        "compact_separator",
    )

    # 2023 CK control bridge confirmed by diagnostic
    check(
        "CK-2-1",
        ["721", "722", "723", "724", "725", "726"],
        "721",
        "ck_to_7prefix",
    )

    check(
        "CK-2-6",
        ["721", "722", "723", "724", "725", "726"],
        "726",
        "ck_to_7prefix",
    )

    # 2024 exact IDs
    check(
        "L111",
        ["L111", "L112", "CK11"],
        "L111",
        "exact",
    )

    check(
        "CK11",
        ["L111", "L112", "CK11"],
        "CK11",
        "exact",
    )

    # 2025 segmented production ID -> growth position ID
    check(
        "1-1-1",
        ["L111", "L112", "L113"],
        "L111",
        "year2025_L_position",
        year=2025,
    )

    check(
        "2-1-4",
        ["L214", "L215", "L216"],
        "L214",
        "year2025_L_position",
        year=2025,
    )

    # Ambiguity safeguard: numeric mapping must not guess.
    idx = build_growth_identity_index(
        ["L111", "X111"]
    )
    got, how = harmonize_target_id(
        "1-1-1",
        idx,
    )
    assert got is None
    assert how == "unmatched"
    print(
        "PASS: ambiguous numeric core is rejected."
    )

    print(
        "\nIDENTITY HARMONIZATION TEST PASSED"
    )

if __name__ == "__main__":
    main()
