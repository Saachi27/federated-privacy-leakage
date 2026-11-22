from scripts import analyze_addons as aa


def test_every_comparison_looks_up_its_own_setting():
    for group, name, setting, n, pc, single_fn, multi_fn in aa.comparisons():
        assert pc[0] == setting and pc[1] == n
        for kind in aa.KINDS:
            for key in (single_fn(kind), multi_fn(kind)):
                assert key[0] == setting, (group, name, key)
                assert key[1] == n and key[4] == kind


def test_e10_rows_use_e10_and_cifar_rows_use_e5():
    rows = {c[0]: c for c in aa.comparisons() if c[0] in ("21.B", "21.C")}
    assert rows["21.B"][5]("cosine") == ("mnist-fresh", 2, 10, "1x2", "cosine")
    assert rows["21.C"][6]("cosine") == ("cifar10-fresh", 2, 5, "2x1", "cosine")
