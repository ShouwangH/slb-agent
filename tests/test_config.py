"""Tests for EngineConfig and default configuration."""

import pytest

from app.config import DEFAULT_CAP_RATE_CURVE, DEFAULT_ENGINE_CONFIG, EngineConfig
from app.models import AssetType, MarketTier


class TestDefaultCapRateCurve:
    """Tests for DEFAULT_CAP_RATE_CURVE constant."""

    def test_all_asset_types_present(self):
        """Every AssetType has cap rates defined."""
        for asset_type in AssetType:
            assert asset_type in DEFAULT_CAP_RATE_CURVE

    def test_all_market_tiers_present(self):
        """Every AssetType has all MarketTier values."""
        for asset_type in AssetType:
            for tier in MarketTier:
                assert tier in DEFAULT_CAP_RATE_CURVE[asset_type]

    def test_cap_rates_are_valid(self):
        """Cap rates are positive and less than 1."""
        for asset_type in AssetType:
            for tier in MarketTier:
                rate = DEFAULT_CAP_RATE_CURVE[asset_type][tier]
                assert 0 < rate < 1, f"Invalid cap rate {rate} for {asset_type}/{tier}"

    def test_tier_ordering(self):
        """Higher tiers (less liquid) have higher cap rates."""
        for asset_type in AssetType:
            rates = DEFAULT_CAP_RATE_CURVE[asset_type]
            assert rates[MarketTier.TIER_1] < rates[MarketTier.TIER_2]
            assert rates[MarketTier.TIER_2] < rates[MarketTier.TIER_3]

    def test_specific_values_match_design(self):
        """Spot-check specific values from DESIGN.md Section 5.1."""
        assert DEFAULT_CAP_RATE_CURVE[AssetType.STORE][MarketTier.TIER_1] == 0.055
        assert DEFAULT_CAP_RATE_CURVE[AssetType.DISTRIBUTION_CENTER][MarketTier.TIER_1] == 0.045
        assert DEFAULT_CAP_RATE_CURVE[AssetType.OFFICE][MarketTier.TIER_2] == 0.070


class TestEngineConfig:
    """Tests for EngineConfig model."""

    def test_default_values(self):
        """Default config matches DESIGN.md Section 5.1."""
        config = EngineConfig()
        assert config.default_market_tier == MarketTier.TIER_2
        assert config.transaction_haircut == 0.025
        assert config.avg_cost_of_debt == 0.06
        assert config.slb_rent_multiplier == 1.0
        assert config.target_tolerance == 0.02
        assert config.criticality_threshold == 0.7
        assert config.epsilon == 1e-9
        assert config.default_max_net_leverage == 4.0
        assert config.default_min_fixed_charge_coverage == 3.0
        assert config.default_min_interest_coverage is None
        assert config.default_max_iterations == 3

    def test_custom_values(self):
        """Config accepts custom values."""
        config = EngineConfig(
            transaction_haircut=0.03,
            avg_cost_of_debt=0.07,
            slb_rent_multiplier=1.1,
            default_max_net_leverage=5.0,
        )
        assert config.transaction_haircut == 0.03
        assert config.avg_cost_of_debt == 0.07
        assert config.slb_rent_multiplier == 1.1
        assert config.default_max_net_leverage == 5.0

    def test_epsilon_must_be_positive(self):
        """Epsilon must be > 0."""
        with pytest.raises(ValueError):
            EngineConfig(epsilon=0)
        with pytest.raises(ValueError):
            EngineConfig(epsilon=-1e-9)

    def test_transaction_haircut_bounds(self):
        """Transaction haircut must be in [0, 1)."""
        # Valid at boundaries
        EngineConfig(transaction_haircut=0)
        EngineConfig(transaction_haircut=0.5)

        # Invalid
        with pytest.raises(ValueError):
            EngineConfig(transaction_haircut=-0.01)
        with pytest.raises(ValueError):
            EngineConfig(transaction_haircut=1.0)

    def test_avg_cost_of_debt_bounds(self):
        """Cost of debt must be in (0, 1)."""
        EngineConfig(avg_cost_of_debt=0.01)
        EngineConfig(avg_cost_of_debt=0.99)

        with pytest.raises(ValueError):
            EngineConfig(avg_cost_of_debt=0)
        with pytest.raises(ValueError):
            EngineConfig(avg_cost_of_debt=1.0)

    def test_slb_rent_multiplier_must_be_positive(self):
        """SLB rent multiplier must be > 0."""
        EngineConfig(slb_rent_multiplier=0.5)
        EngineConfig(slb_rent_multiplier=2.0)

        with pytest.raises(ValueError):
            EngineConfig(slb_rent_multiplier=0)
        with pytest.raises(ValueError):
            EngineConfig(slb_rent_multiplier=-0.1)

    def test_criticality_threshold_bounds(self):
        """Criticality threshold must be in [0, 1]."""
        EngineConfig(criticality_threshold=0)
        EngineConfig(criticality_threshold=1)

        with pytest.raises(ValueError):
            EngineConfig(criticality_threshold=-0.1)
        with pytest.raises(ValueError):
            EngineConfig(criticality_threshold=1.1)

    def test_max_iterations_bounds(self):
        """Max iterations must be in [1, 10]."""
        EngineConfig(default_max_iterations=1)
        EngineConfig(default_max_iterations=10)

        with pytest.raises(ValueError):
            EngineConfig(default_max_iterations=0)
        with pytest.raises(ValueError):
            EngineConfig(default_max_iterations=11)


class TestDefaultEngineConfig:
    """Tests for DEFAULT_ENGINE_CONFIG instance."""

    def test_is_engine_config(self):
        """DEFAULT_ENGINE_CONFIG is an EngineConfig instance."""
        assert isinstance(DEFAULT_ENGINE_CONFIG, EngineConfig)

    def test_has_cap_rate_curve(self):
        """DEFAULT_ENGINE_CONFIG has the cap rate curve."""
        assert DEFAULT_ENGINE_CONFIG.cap_rate_curve is not None
        assert len(DEFAULT_ENGINE_CONFIG.cap_rate_curve) == len(AssetType)

    def test_serialization_roundtrip(self):
        """Config can be serialized and deserialized."""
        json_str = DEFAULT_ENGINE_CONFIG.model_dump_json()
        restored = EngineConfig.model_validate_json(json_str)

        assert restored.transaction_haircut == DEFAULT_ENGINE_CONFIG.transaction_haircut
        assert restored.avg_cost_of_debt == DEFAULT_ENGINE_CONFIG.avg_cost_of_debt
        assert restored.epsilon == DEFAULT_ENGINE_CONFIG.epsilon
