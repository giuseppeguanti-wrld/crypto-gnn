"""Tests for cryptognn.config: the single source of truth, and its fingerprint.

`config/default.yaml` holds every parameter of the study, so `load_config()`
turning it into typed dataclasses is what lets the rest of the codebase contain
no magic numbers. The tests here read the **real** config rather than a fixture:
the point is not that the parser works on some YAML, but that the shipped file
still parses into the shape the code expects, and that the frozen grid is still
the frozen grid.

`config_hash()` has no caller yet -- S5.3 writes it into the run manifest -- and
that is exactly why it is tested now. An unused function with no contract drifts
until the day it is needed and then cannot be trusted for the one thing it is
for: telling whether two sets of results came from the same configuration.
"""
from __future__ import annotations

from datetime import date

import pytest
import yaml

from cryptognn.config import Config, config_hash, load_config
from cryptognn.paths import DEFAULT_CONFIG


@pytest.fixture
def config() -> Config:
    return load_config(DEFAULT_CONFIG)


class TestLoadConfig:
    def test_parses_the_shipped_config_into_nested_dataclasses(self, config):
        assert isinstance(config, Config)
        assert isinstance(config.data.start, date) and isinstance(config.data.end, date)
        assert config.data.start < config.data.end
        assert config.graph.threshold.method == "permutation"
        assert config.walkforward.mode in ("rolling", "expanding")

    def test_the_universe_is_the_fifteen_frozen_assets(self, config):
        assert len(config.data.symbols) == 15
        assert len(set(config.data.symbols)) == 15, "a duplicated symbol would silently shrink the universe"
        assert config.data.symbols[0] == "BTC"

    def test_the_grid_frozen_in_sprint_1_is_still_four_configurations(self, config):
        """The count is reported in the thesis as part of the protocol. If this
        test fails, either the grid was reopened -- which the end-of-Sprint-4 rule
        forbids -- or the number written in Section 6.4 is now wrong.
        """
        grid = config.model.gcn

        assert len(grid.hidden) * len(grid.dropout) == 4
        assert len(grid.seeds) == 5
        assert grid.epochs > 0 and grid.patience > 0

    def test_the_offset_relation_the_harness_depends_on_holds(self, config):
        """make_folds_from_config() derives its offset as graph.window - 1, and
        the feature warm-up must fit inside it or every fold would start on NaN.
        """
        assert config.graph.window > max(config.features.vol_windows)
        assert config.graph.window > config.features.lags

    def test_an_unknown_key_is_refused_rather_than_ignored(self, tmp_path):
        """A typo in the YAML must fail loudly. Silently dropping an unrecognized
        key is how a study ends up running with a parameter nobody set.
        """
        with DEFAULT_CONFIG.open() as f:
            raw = yaml.safe_load(f)
        raw["backtest"]["cost_bpz"] = 10
        broken = tmp_path / "typo.yaml"
        broken.write_text(yaml.safe_dump(raw))

        with pytest.raises(TypeError):
            load_config(broken)


class TestConfigHash:
    def test_is_stable_across_reads(self):
        assert config_hash(DEFAULT_CONFIG) == config_hash(DEFAULT_CONFIG)

    def test_ignores_key_order_and_formatting(self, tmp_path):
        """Two files that say the same thing must fingerprint the same, or the
        hash would report a configuration change every time the YAML is tidied.
        """
        with DEFAULT_CONFIG.open() as f:
            raw = yaml.safe_load(f)

        reordered = tmp_path / "reordered.yaml"
        reordered.write_text(yaml.safe_dump(dict(reversed(list(raw.items()))), sort_keys=False, indent=4))

        assert config_hash(reordered) == config_hash(DEFAULT_CONFIG)

    def test_changes_when_any_value_changes(self, tmp_path):
        """The property the manifest depends on: a different study is a different
        hash, including for a parameter buried three levels down.
        """
        with DEFAULT_CONFIG.open() as f:
            raw = yaml.safe_load(f)
        raw["graph"]["threshold"]["alpha"] = 0.01
        altered = tmp_path / "altered.yaml"
        altered.write_text(yaml.safe_dump(raw))

        assert config_hash(altered) != config_hash(DEFAULT_CONFIG)

    def test_is_a_sha1_hex_digest(self):
        fingerprint = config_hash(DEFAULT_CONFIG)

        assert len(fingerprint) == 40
        assert set(fingerprint) <= set("0123456789abcdef")
