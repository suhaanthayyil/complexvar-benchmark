from complexvar.features import interface_burial_proxy, mutation_descriptor


def test_mutation_descriptor_has_expected_deltas():
    descriptor = mutation_descriptor("A", "D")
    assert descriptor["delta_charge"] < 0
    assert descriptor["delta_hydrophobicity"] < 0
    assert descriptor["changed_to_gly"] == 0


def test_interface_burial_proxy_increases_with_contacts():
    low = interface_burial_proxy(
        local_degree=2, inter_chain_contacts=1, solvent_proxy=5
    )
    high = interface_burial_proxy(
        local_degree=5, inter_chain_contacts=4, solvent_proxy=2
    )
    assert high > low
