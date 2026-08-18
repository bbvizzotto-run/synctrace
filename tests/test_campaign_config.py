"""Tests guarding the official experimental protocol configuration."""

from pathlib import Path

import yaml


def test_campaign_config_has_three_seeds_and_identity_splits():
    config = yaml.safe_load(Path("config/campaign.yaml").read_text())
    assert len(config["seeds"]) == 3
    assert sum(config["splits"].values()) == 1.0
    assert {"fixed_severity", "no_mamba", "dense_attention", "no_sae"}.issubset(config["ablations"])
