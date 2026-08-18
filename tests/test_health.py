from orchestrate_agent_lab.health import health_check


def test_health_check():
    assert health_check() == {"status": "ok"}
