import pytest

from src.tokenization.cost import ModelPricing, calculate_cost


def test_calculates_input_and_output_costs():
    pricing = ModelPricing(
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    )

    cost = calculate_cost(
        input_tokens=500_000,
        output_tokens=100_000,
        pricing=pricing,
    )

    assert cost == pytest.approx(2.25)


def test_calculates_input_only_cost():
    pricing = ModelPricing(
        input_cost_per_million=3.00,
        output_cost_per_million=12.00,
    )

    assert calculate_cost(250_000, 0, pricing) == pytest.approx(0.75)


def test_returns_zero_for_zero_usage():
    pricing = ModelPricing(
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    )

    assert calculate_cost(0, 0, pricing) == 0.0


@pytest.mark.parametrize(
    ("input_tokens", "output_tokens"),
    [(-1, 0), (0, -1)],
)
def test_rejects_negative_token_counts(input_tokens, output_tokens):
    pricing = ModelPricing(
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    )

    with pytest.raises(ValueError, match="Invalid token count"):
        calculate_cost(input_tokens, output_tokens, pricing)


@pytest.mark.parametrize(
    "pricing",
    [
        ModelPricing(input_cost_per_million=-0.01, output_cost_per_million=10.00),
        ModelPricing(input_cost_per_million=2.50, output_cost_per_million=-0.01),
    ],
)
def test_rejects_negative_prices(pricing):
    with pytest.raises(ValueError, match="Invalid cost"):
        calculate_cost(1, 1, pricing)


def test_does_not_round_the_calculated_cost():
    pricing = ModelPricing(
        input_cost_per_million=1.0,
        output_cost_per_million=0.0,
    )

    assert calculate_cost(1, 0, pricing) == pytest.approx(0.000001)


def test_pricing_is_immutable():
    pricing = ModelPricing(
        input_cost_per_million=2.50,
        output_cost_per_million=10.00,
    )

    with pytest.raises(AttributeError):
        pricing.input_cost_per_million = 3.00
