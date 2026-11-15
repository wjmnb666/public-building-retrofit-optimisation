"""Authoritative annual state-transition kernel for exact and DQN solvers.

The kernel contains every financial and physical-carbon equation used by the
sequential experiment.  It has no dependency on exact-solver result files,
preferred policies, pathway rankings or learned models.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Mapping, Sequence

import numpy as np


HOLD = 0
UPGRADE_LEVELS = (1, 2, 3)
ACTION_NAMES = ("hold", "upgrade_to_level_1", "upgrade_to_level_2", "upgrade_to_level_3")


@dataclass(frozen=True)
class AnnualState:
    """Physical and financial state before the current year's budget inflow."""

    year: int
    current_level: int
    electricity_kwh_m2yr: float
    gas_kwh_m2yr: float
    available_budget_gbp: float
    layer_ages_years: tuple[int, int, int] = (0, 0, 0)
    layer_service_lives_years: tuple[int, int, int] = (0, 0, 0)
    layer_replacement_costs_gbp: tuple[float, float, float] = (0.0, 0.0, 0.0)
    layer_replacement_embodied_kgco2e: tuple[float, float, float] = (0.0, 0.0, 0.0)
    cumulative_discounted_lcc_gbp: float = 0.0
    cumulative_physical_carbon_kgco2e: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "layer_ages_years",
            "layer_service_lives_years",
            "layer_replacement_costs_gbp",
            "layer_replacement_embodied_kgco2e",
        ):
            if len(getattr(self, name)) != 3:
                raise ValueError(f"{name} must contain exactly three incremental layers")

    @property
    def package_age_years(self) -> int:
        """Compatibility view: age of the highest installed layer."""

        return 0 if self.current_level <= 0 else int(self.layer_ages_years[self.current_level - 1])

    @property
    def effective_service_life_years(self) -> int:
        """Compatibility view: service life of the highest installed layer."""

        return 0 if self.current_level <= 0 else int(self.layer_service_lives_years[self.current_level - 1])

    @property
    def replacement_cost_gbp(self) -> float:
        return float(sum(self.layer_replacement_costs_gbp[: self.current_level]))

    @property
    def replacement_embodied_kgco2e(self) -> float:
        return float(sum(self.layer_replacement_embodied_kgco2e[: self.current_level]))


@dataclass(frozen=True)
class AnnualExogenous:
    """Year-specific prices and physical carbon factors."""

    calendar_year: int
    electricity_price_gbp_kwh: float
    gas_price_gbp_kwh: float
    electricity_carbon_kgco2e_kwh: float
    gas_carbon_kgco2e_kwh: float


@dataclass(frozen=True)
class TransitionResult:
    """Complete auditable result of one annual transition."""

    next_state: AnnualState
    financial: dict[str, float]
    carbon: dict[str, float]
    vector_reward: tuple[float, float]
    scalar_reward: float
    action_mask: tuple[bool, bool, bool, bool]
    selected_action: int
    applied_action: int
    feasible: bool
    diagnostics: dict[str, Any]


def package_table(
    accepted: Mapping[str, Any],
    area_m2: float,
    *,
    embodied_case: str = "central",
    service_life_offset: int = 0,
    capex_multiplier: float = 1.0,
    savings_multiplier: float = 1.0,
    embodied_targets_override: Mapping[int, float] | None = None,
) -> dict[int, dict[str, float]]:
    """Build absolute target-level parameters; level differences are calculated later."""

    table: dict[int, dict[str, float]] = {
        0: {
            "capex_gbp": 0.0,
            "embodied_kgco2e": 0.0,
            "electricity_saving_fraction": 0.0,
            "gas_saving_fraction": 0.0,
            "service_life_years": 0.0,
        }
    }
    for level in UPGRADE_LEVELS:
        source = accepted["packages"][f"PK{level}"]
        embodied_m2 = (
            float(embodied_targets_override[level])
            if embodied_targets_override is not None
            else float(source["embodied_kgco2e_m2"][embodied_case])
        )
        table[level] = {
            "capex_gbp": float(source["total_capex_gbp_m2"]) * area_m2 * capex_multiplier,
            "embodied_kgco2e": embodied_m2 * area_m2,
            "electricity_saving_fraction": min(
                1.0, float(source["electricity_saving_fraction"]) * savings_multiplier
            ),
            "gas_saving_fraction": min(1.0, float(source["gas_saving_fraction"]) * savings_multiplier),
            "service_life_years": float(
                max(1, int(source["effective_service_life_years"]) + int(service_life_offset))
            ),
        }
    return table


def annual_budget_inflow(
    scenario: Mapping[str, Any], packages: Mapping[int, Mapping[str, float]]
) -> float:
    """Return annual cash budget; unconstrained cases receive a non-binding full-PK3 inflow."""

    ratio = scenario.get("budget_ratio")
    return float(packages[3]["capex_gbp"] * (1.0 if ratio is None else float(ratio)))


def stranded_cost(
    year: int,
    incremental_capex_gbp: float,
    natural_window_years: int | None,
    coefficient: float,
) -> float:
    """One-off natural-window opportunity cost, including Year 0."""

    if natural_window_years is None:
        return 0.0
    window = int(natural_window_years)
    if window <= 0:
        raise ValueError("natural replacement window must be positive")
    remaining_life = max(window - int(year), 0)
    return float(coefficient) * float(incremental_capex_gbp) * remaining_life / window


def feasibility_mask(
    state: AnnualState,
    scenario: Mapping[str, Any],
    packages: Mapping[int, Mapping[str, float]],
    *,
    include_current_year_inflow: bool = True,
) -> tuple[bool, bool, bool, bool]:
    """Return the monotonic and annual-cash-budget action mask."""

    budget = float(state.available_budget_gbp)
    if include_current_year_inflow:
        budget += annual_budget_inflow(scenario, packages)
    mask = [True]
    for target in UPGRADE_LEVELS:
        increment = float(packages[target]["capex_gbp"] - packages[state.current_level]["capex_gbp"])
        mask.append(target > state.current_level and increment <= budget + 1e-9)
    return tuple(bool(value) for value in mask)


def initial_state(
    baseline: Mapping[str, float],
) -> AnnualState:
    """Create the pre-Year-0 state directly from a prototype baseline."""

    return AnnualState(
        year=0,
        current_level=0,
        electricity_kwh_m2yr=float(baseline["electricity_kwh_m2yr"]),
        gas_kwh_m2yr=float(baseline["gas_kwh_m2yr"]),
        available_budget_gbp=0.0,
    )


def incremental_layer_value(
    packages: Mapping[int, Mapping[str, float]], level: int, field: str
) -> float:
    """Return one nested layer's target-level difference."""

    value = float(packages[level][field] - packages[level - 1][field])
    if value < -1e-9:
        raise ValueError(f"Nested package {field} decreases at level {level}")
    return max(value, 0.0)


def replacement_embodied_factor(
    year: int,
    horizon: int,
    scenario: Mapping[str, Any],
    packages: Mapping[int, Mapping[str, float]],
    level: int,
) -> float:
    """Return a factor for constant or central-to-low planning-envelope sensitivity.

    The declining case adds no external numeric parameter: it linearly connects
    the registered initial-case and low-envelope target values over the declared
    appraisal horizon.  It is a stylised prospective sensitivity, not observed
    Manchester evidence or an EPD forecast.
    """

    rule = str(scenario.get("replacement_embodied_rule", "constant"))
    if rule == "constant":
        return 1.0
    if rule != "linear_to_low_envelope":
        raise ValueError(f"Unsupported replacement embodied-carbon rule: {rule}")
    low_targets = scenario.get("replacement_low_targets_kgco2e")
    if low_targets is None:
        raise ValueError("Declining replacement embodied-carbon rule lacks registered low targets")
    current = incremental_layer_value(packages, level, "embodied_kgco2e")
    if current <= 0:
        return 1.0
    low = float(low_targets[level]) - float(low_targets.get(level - 1, 0.0))
    fraction = min(max(float(year) / max(int(horizon) - 1, 1), 0.0), 1.0)
    replacement_value = current + fraction * (low - current)
    return max(replacement_value / current, 0.0)


def transition(
    state: AnnualState,
    selected_action: int,
    scenario: Mapping[str, Any],
    baseline: Mapping[str, float],
    exogenous: AnnualExogenous,
    packages: Mapping[int, Mapping[str, float]],
    *,
    horizon: int,
    discount_rate: float,
    objective_scales: Sequence[float] = (1.0, 1.0),
    objective_weights: Sequence[float] = (0.5, 0.5),
) -> TransitionResult:
    """Apply the documented beginning-of-year action and same-year operation."""

    if not 0 <= int(selected_action) <= 3:
        raise ValueError("action must be in 0..3")
    if state.year >= int(horizon):
        raise RuntimeError("transition requested after the appraisal horizon")

    inflow = annual_budget_inflow(scenario, packages)
    budget_after_inflow = float(state.available_budget_gbp + inflow)
    action_mask = feasibility_mask(state, scenario, packages, include_current_year_inflow=True)
    feasible = bool(action_mask[int(selected_action)])
    applied_action = int(selected_action) if feasible else HOLD
    upgraded = applied_action > state.current_level

    current_level = int(state.current_level)
    layer_ages = list(map(int, state.layer_ages_years))
    layer_lives = list(map(int, state.layer_service_lives_years))
    layer_cost_bases = list(map(float, state.layer_replacement_costs_gbp))
    layer_carbon_bases = list(map(float, state.layer_replacement_embodied_kgco2e))
    installed_this_year: set[int] = set()
    capex = upfront = opportunity_cost = 0.0
    if upgraded:
        target = int(applied_action)
        capex = float(packages[target]["capex_gbp"] - packages[current_level]["capex_gbp"])
        upfront = float(
            packages[target]["embodied_kgco2e"] - packages[current_level]["embodied_kgco2e"]
        )
        opportunity_cost = stranded_cost(
            state.year,
            capex,
            scenario.get("replacement_window"),
            float(scenario.get("stranded_value_coefficient", 0.0)),
        )
        budget_after_inflow -= capex
        for layer in range(current_level + 1, target + 1):
            index = layer - 1
            installed_this_year.add(layer)
            layer_ages[index] = 0
            layer_lives[index] = int(packages[layer]["service_life_years"])
            layer_cost_bases[index] = incremental_layer_value(packages, layer, "capex_gbp")
            layer_carbon_bases[index] = incremental_layer_value(
                packages, layer, "embodied_kgco2e"
            )
        current_level = target

    electricity_intensity = float(baseline["electricity_kwh_m2yr"]) * (
        1.0 - float(packages[current_level]["electricity_saving_fraction"])
    )
    gas_intensity = float(baseline["gas_kwh_m2yr"]) * (
        1.0 - float(packages[current_level]["gas_saving_fraction"])
    )
    area = float(baseline["floor_area_m2"])
    electricity_kwh = electricity_intensity * area
    gas_kwh = gas_intensity * area
    electricity_cost = electricity_kwh * float(exogenous.electricity_price_gbp_kwh)
    gas_cost = gas_kwh * float(exogenous.gas_price_gbp_kwh)
    maintenance = (
        float(scenario.get("maintenance_fraction", 0.0))
        * float(packages[current_level]["capex_gbp"])
        if current_level > 0
        else 0.0
    )
    operational_electricity = electricity_kwh * float(exogenous.electricity_carbon_kgco2e_kwh)
    operational_gas = gas_kwh * float(exogenous.gas_carbon_kgco2e_kwh)

    scheduled_replacement_layers: list[int] = []
    replacement_cost_by_layer = [0.0, 0.0, 0.0]
    replacement_carbon_by_layer = [0.0, 0.0, 0.0]
    next_layer_ages = [0, 0, 0]
    for layer in range(1, current_level + 1):
        index = layer - 1
        if layer in installed_this_year:
            # Beginning-of-year installation supplies one full year of service.
            next_layer_ages[index] = 1
        elif (
            layer_ages[index] + 1 >= layer_lives[index]
            and state.year < int(horizon) - 1
        ):
            scheduled_replacement_layers.append(layer)
            replacement_cost_by_layer[index] = layer_cost_bases[index]
            replacement_carbon_by_layer[index] = (
                layer_carbon_bases[index]
                * replacement_embodied_factor(state.year, horizon, scenario, packages, layer)
            )
            # Replacement occurs after the year's service in the registered
            # event order, so the renewed layer is age zero at year end.  A
            # final-boundary replacement is not installed because it would
            # provide no subsequent operational year inside the appraisal.
            next_layer_ages[index] = 0
        else:
            next_layer_ages[index] = layer_ages[index] + 1
    scheduled_replacement = bool(scheduled_replacement_layers)
    replacement_cost = float(sum(replacement_cost_by_layer))
    replacement_carbon = float(sum(replacement_carbon_by_layer))

    residual_by_layer = [0.0, 0.0, 0.0]
    terminal = state.year == int(horizon) - 1
    residual_enabled = bool(scenario.get("terminal_residual_enabled", True))
    if terminal and residual_enabled:
        for layer in range(1, current_level + 1):
            index = layer - 1
            life = layer_lives[index]
            if life > 0:
                residual_by_layer[index] = (
                    layer_cost_bases[index]
                    * max(life - next_layer_ages[index], 0)
                    / life
                )
    residual = float(sum(residual_by_layer))
    new_layer_residual = float(
        sum(residual_by_layer[layer - 1] for layer in installed_this_year)
    )

    undiscounted = (
        capex
        + opportunity_cost
        + electricity_cost
        + gas_cost
        + maintenance
        + replacement_cost
        - residual
    )
    discount_factor = 1.0 / ((1.0 + float(discount_rate)) ** (state.year + 1))
    discounted = undiscounted * discount_factor
    physical_carbon = operational_electricity + operational_gas + upfront + replacement_carbon
    cost_scale = max(float(objective_scales[0]), 1e-12)
    carbon_scale = max(float(objective_scales[1]), 1e-12)
    cost_weight, carbon_weight = map(float, objective_weights)
    vector_reward = (-discounted, -physical_carbon)
    scalar_reward = cost_weight * vector_reward[0] / cost_scale + carbon_weight * vector_reward[1] / carbon_scale

    next_state = AnnualState(
        year=state.year + 1,
        current_level=current_level,
        electricity_kwh_m2yr=electricity_intensity,
        gas_kwh_m2yr=gas_intensity,
        available_budget_gbp=max(budget_after_inflow, 0.0),
        layer_ages_years=tuple(next_layer_ages),
        layer_service_lives_years=tuple(layer_lives),
        layer_replacement_costs_gbp=tuple(layer_cost_bases),
        layer_replacement_embodied_kgco2e=tuple(layer_carbon_bases),
        cumulative_discounted_lcc_gbp=state.cumulative_discounted_lcc_gbp + discounted,
        cumulative_physical_carbon_kgco2e=state.cumulative_physical_carbon_kgco2e + physical_carbon,
    )
    financial = {
        "budget_inflow_gbp": inflow,
        "available_budget_before_action_gbp": state.available_budget_gbp + inflow,
        "capital_expenditure_gbp": capex,
        "stranded_value_gbp": opportunity_cost,
        "electricity_expenditure_gbp": electricity_cost,
        "gas_expenditure_gbp": gas_cost,
        "maintenance_gbp": maintenance,
        "replacement_cost_gbp": replacement_cost,
        "replacement_cost_layer_1_gbp": replacement_cost_by_layer[0],
        "replacement_cost_layer_2_gbp": replacement_cost_by_layer[1],
        "replacement_cost_layer_3_gbp": replacement_cost_by_layer[2],
        "residual_value_gbp_credit": residual,
        "residual_value_layer_1_gbp_credit": residual_by_layer[0],
        "residual_value_layer_2_gbp_credit": residual_by_layer[1],
        "residual_value_layer_3_gbp_credit": residual_by_layer[2],
        "newly_installed_layer_residual_gbp_credit": new_layer_residual,
        "net_new_investment_after_residual_gbp": capex - new_layer_residual,
        "undiscounted_cashflow_gbp": undiscounted,
        "discount_factor": discount_factor,
        "discounted_cashflow_gbp": discounted,
        "remaining_budget_gbp": next_state.available_budget_gbp,
    }
    carbon = {
        "operational_electricity_kgco2e": operational_electricity,
        "operational_gas_kgco2e": operational_gas,
        "upfront_embodied_kgco2e": upfront,
        "replacement_carbon_kgco2e": replacement_carbon,
        "replacement_carbon_layer_1_kgco2e": replacement_carbon_by_layer[0],
        "replacement_carbon_layer_2_kgco2e": replacement_carbon_by_layer[1],
        "replacement_carbon_layer_3_kgco2e": replacement_carbon_by_layer[2],
        "annual_total_included_wlc_kgco2e": physical_carbon,
    }
    diagnostics = {
        "event_order_version": "direct_annual_component_layers_v3",
        "selected_action_name": ACTION_NAMES[int(selected_action)],
        "applied_action_name": ACTION_NAMES[int(applied_action)],
        "infeasible_action_replaced_by_hold": not feasible,
        "upgraded": upgraded,
        "scheduled_replacement": scheduled_replacement,
        "scheduled_replacement_layers": tuple(scheduled_replacement_layers),
        "installed_layers_this_year": tuple(sorted(installed_this_year)),
        "terminal_residual_applied": terminal and residual > 0,
        "terminal_residual_enabled": residual_enabled,
        "natural_replacement_window_reached": bool(
            scenario.get("replacement_window") is not None
            and state.year >= int(scenario["replacement_window"])
        ),
    }
    return TransitionResult(
        next_state=next_state,
        financial=financial,
        carbon=carbon,
        vector_reward=vector_reward,
        scalar_reward=float(scalar_reward),
        action_mask=action_mask,
        selected_action=int(selected_action),
        applied_action=int(applied_action),
        feasible=feasible,
        diagnostics=diagnostics,
    )


def state_as_dict(state: AnnualState) -> dict[str, Any]:
    """Serialise a state without hiding cumulative reporting ledgers."""

    return asdict(state)
