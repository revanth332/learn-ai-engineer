from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPricing:
    input_cost_per_million: float
    output_cost_per_million: float


def calculate_cost(
    input_tokens: int,
    output_tokens: int,
    pricing: ModelPricing,
) -> float:
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Invalid token count")
    if pricing.input_cost_per_million < 0 or pricing.output_cost_per_million < 0:
        raise ValueError("Invalid cost")
    cost = (input_tokens*pricing.input_cost_per_million)/10_00_000 + (output_tokens*pricing.output_cost_per_million)/10_00_000
    return cost