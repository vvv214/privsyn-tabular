from method.synthesis.privsyn.lib_dataset.domain import Domain


def test_domain_core_operations():
    domain = Domain(["age", "income", "city"], [10, 5, 3])

    # change_shape should update config and shape tuple
    domain.change_shape("income", 7)
    assert domain["income"] == 7
    assert domain.shape == (10, 7, 3)

    # project accepts strings and returns new Domain
    proj_single = domain.project("city")
    assert proj_single.attrs == ("city",)
    assert proj_single.shape == (3,)

    proj_subset = domain.project(["income", "age"])
    assert proj_subset.attrs == ("income", "age")
    assert proj_subset.shape == (7, 10)

    # marginalize removes columns
    marginalized = domain.marginalize(["income"])
    assert marginalized.attrs == ("age", "city")

    # axes and transpose
    assert domain.axes(["age", "city"]) == (0, 2)
    assert domain.transpose(["city", "age"]).attrs == ("city", "age")

    # invert and contains
    assert domain.invert(["age"]) == ["income", "city"]
    other = Domain(["income"], [7])
    assert domain.contains(other)

    # merge with overlapping attributes keeps unique order
    merged = domain.merge(Domain(["city", "state"], [3, 4]))
    assert merged.attrs == ("age", "income", "city", "state")
    assert merged.shape == (10, 7, 3, 4)

    # size calculations (full and subset)
    assert domain.size() == 10 * 7 * 3
    assert domain.size(["income", "city"]) == 7 * 3

    # sorting and canonical ordering
    by_size = domain.sort("size")
    assert by_size.attrs[0] == "city"
    by_name = domain.sort("name")
    assert by_name.attrs == ("age", "city", "income")
    assert domain.canonical(["city", "age", "missing"]) == ("age", "city")

    # equality and representation helpers
    assert domain == Domain.fromdict({"age": 10, "income": 7, "city": 3})
    repr_text = repr(domain)
    assert "Domain(" in repr_text and "age" in repr_text


def test_domain_iter_protocol():
    domain = Domain(["a", "b"], [2, 3])
    attrs = list(iter(domain))
    assert attrs == ["a", "b"]
    assert len(domain) == 2
    assert "a" in domain
    assert "c" not in domain
