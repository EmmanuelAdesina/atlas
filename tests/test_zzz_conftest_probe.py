import atlas.ops.campaign as campaign


def test_conftest_probe_state_file_is_proven():
    print("STATE_FILE:", campaign.STATE_FILE)
    txt = campaign.STATE_FILE.read_text(encoding="utf-8")
    print("content:", txt)
    assert "proven" in txt
