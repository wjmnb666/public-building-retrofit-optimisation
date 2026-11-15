"""Direct finite-horizon LCC and planning-level WLC pathway evaluation."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any

import numpy as np
import pandas as pd

from .data_sources import PROTOTYPES, load_config
from .transition_kernel import (
    AnnualExogenous,
    feasibility_mask,
    initial_state,
    package_table,
    state_as_dict,
    transition,
)


@dataclass(frozen=True)
class Pathway:
    """Strictly monotonic start-of-year upgrade schedule."""

    actions: tuple[tuple[int, int], ...]

    @property
    def pathway_id(self) -> str:
        return "no_action" if not self.actions else "-".join(
            f"L{level}@{year}" for year, level in self.actions
        )

    @property
    def first_action_year(self) -> int | None:
        return self.actions[0][0] if self.actions else None

    @property
    def final_level(self) -> int:
        return self.actions[-1][1] if self.actions else 0


def enumerate_pathways(horizon: int = 30) -> list[Pathway]:
    """Enumerate every feasible-form monotonic schedule without downgrades."""

    years = range(horizon)
    paths = [Pathway(())]
    level_sequences = [(1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
    for levels in level_sequences:
        for selected_years in combinations(years, len(levels)):
            paths.append(Pathway(tuple(zip(selected_years, levels))))
    return paths


def _scenario(
    name: str,
    family: str,
    *,
    budget_ratio: float | None = None,
    replacement_window: int | None = None,
    stranded_value_coefficient: float = 0.0,
    grid_case: str = "grid_average",
    embodied_case: str = "central",
    maintenance_fraction: float = 0.0,
    service_life_offset: int = 0,
    replacement_embodied_rule: str = "constant",
) -> dict[str, Any]:
    return {
        "scenario": name,
        "family": family,
        "budget_ratio": budget_ratio,
        "replacement_window": replacement_window,
        "stranded_value_coefficient": stranded_value_coefficient,
        "grid_case": grid_case,
        "embodied_case": embodied_case,
        "maintenance_fraction": maintenance_fraction,
        "service_life_offset": service_life_offset,
        "replacement_embodied_rule": replacement_embodied_rule,
        "energy_price_multiplier": 1.0,
    }


def scenario_registry() -> list[dict[str, Any]]:
    """Registered direct-transition scenario set.

    Stranded-value coefficients and declining replacement embodied carbon are
    explicitly stylised prospective sensitivities; they are not Manchester
    observations.
    """

    return [
        _scenario("central", "central_deterministic"),
        *[
            _scenario(f"budget_{ratio:.2f}", "annual_capital_budget", budget_ratio=ratio)
            for ratio in (0.15, 0.25, 0.40)
        ],
        *[
            _scenario(
                f"replacement_window_{window}",
                "replacement_cycle_alignment",
                replacement_window=window,
                stranded_value_coefficient=0.50,
            )
            for window in (5, 10, 15)
        ],
        _scenario(
            "replacement_window_10_stranding_low",
            "stranded_value_sensitivity",
            replacement_window=10,
            stranded_value_coefficient=0.25,
        ),
        _scenario(
            "replacement_window_10_stranding_high",
            "stranded_value_sensitivity",
            replacement_window=10,
            stranded_value_coefficient=0.75,
        ),
        _scenario(
            "replacement_window_10_declining_embodied",
            "replacement_embodied_carbon_sensitivity",
            replacement_window=10,
            stranded_value_coefficient=0.50,
            replacement_embodied_rule="linear_to_low_envelope",
        ),
        _scenario(
            "grid_long_run_marginal",
            "electricity_factor_sensitivity",
            grid_case="long_run_marginal",
        ),
        _scenario("embodied_low", "embodied_carbon_envelope", embodied_case="low"),
        _scenario("embodied_high", "embodied_carbon_envelope", embodied_case="high"),
        _scenario("maintenance_0.01", "maintenance_sensitivity", maintenance_fraction=0.01),
        _scenario("maintenance_0.02", "maintenance_sensitivity", maintenance_fraction=0.02),
        _scenario("service_life_minus5", "service_life_sensitivity", service_life_offset=-5),
        _scenario("service_life_plus5", "service_life_sensitivity", service_life_offset=5),
    ]


def prototype_inputs(
    baselines: pd.DataFrame, extracted: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Convert official-source outputs to direct model inputs."""

    result: dict[str, dict[str, Any]] = {}
    grid = extracted["grid_factors"]
    for row in baselines.to_dict(orient="records"):
        prototype = str(row["prototype"])
        result[prototype] = {
            "floor_area_m2": float(row["prototype_floor_area_mean_m2"]),
            "electricity_kwh_m2yr": float(row["calibrated_electricity_kWh_m2yr"]),
            "gas_kwh_m2yr": float(row["calibrated_gas_kWh_m2yr"]),
            "electricity_price_gbp_kwh": float(extracted["electricity_price_gbp_kwh"]),
            "gas_price_gbp_kwh": float(extracted["gas_price_gbp_kwh"]),
            "gas_factor_kgco2e_kwh": float(extracted["gas_factor_kgco2e_kwh_gross_cv"]),
            "epc_operational_emissions_kgco2e_m2yr": float(
                row["epc_area_weighted_emissions_kgCO2e_m2yr"]
            ),
            "grid_average": {
                int(year): float(value)
                for year, value in grid["grid_average_commercial_public"].items()
            },
            "long_run_marginal": {
                int(year): float(value)
                for year, value in grid["long_run_marginal_commercial_public"].items()
            },
            "fuel_case": "stock_average",
        }
    if set(result) != set(PROTOTYPES):
        raise ValueError("Missing prototype baselines")
    return result


def _registered_low_targets(area_m2: float) -> dict[int, float]:
    config = load_config()
    return {
        level: float(config["packages"][f"PK{level}"]["embodied_kgco2e_m2"]["low"])
        * area_m2
        for level in (1, 2, 3)
    }


def package_parameters(
    inputs: dict[str, Any],
    scenario: dict[str, Any],
    *,
    capex_multiplier: float = 1.0,
    savings_multiplier: float = 1.0,
    embodied_targets_override: dict[int, float] | None = None,
) -> dict[int, dict[str, float]]:
    return package_table(
        load_config(),
        float(inputs["floor_area_m2"]),
        embodied_case=str(scenario["embodied_case"]),
        service_life_offset=int(scenario["service_life_offset"]),
        capex_multiplier=capex_multiplier,
        savings_multiplier=savings_multiplier,
        embodied_targets_override=embodied_targets_override,
    )


def year_exogenous(
    inputs: dict[str, Any], scenario: dict[str, Any], year: int, energy_price_multiplier: float = 1.0
) -> AnnualExogenous:
    config = load_config()
    calendar = int(config["project"]["appraisal_start_year"]) + int(year)
    multiplier = float(scenario.get("energy_price_multiplier", 1.0)) * energy_price_multiplier
    return AnnualExogenous(
        calendar_year=calendar,
        electricity_price_gbp_kwh=float(inputs["electricity_price_gbp_kwh"]) * multiplier,
        gas_price_gbp_kwh=float(inputs["gas_price_gbp_kwh"]) * multiplier,
        electricity_carbon_kgco2e_kwh=float(inputs[str(scenario["grid_case"])][calendar]),
        gas_carbon_kgco2e_kwh=float(inputs["gas_factor_kgco2e_kwh"]),
    )


def evaluate_path(
    prototype: str,
    inputs: dict[str, Any],
    path: Pathway,
    scenario: dict[str, Any],
    *,
    capex_multiplier: float = 1.0,
    savings_multiplier: float = 1.0,
    energy_price_multiplier: float = 1.0,
    embodied_targets_override: dict[int, float] | None = None,
    return_ledger: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame | None]:
    """Evaluate a complete path solely through the shared transition kernel."""

    config = load_config()
    horizon = int(config["project"]["horizon_years"])
    discount_rate = float(config["project"]["real_discount_rate"])
    packages = package_parameters(
        inputs,
        scenario,
        capex_multiplier=capex_multiplier,
        savings_multiplier=savings_multiplier,
        embodied_targets_override=embodied_targets_override,
    )
    scenario_run = dict(scenario)
    scenario_run["replacement_low_targets_kgco2e"] = _registered_low_targets(
        float(inputs["floor_area_m2"])
    )
    state = initial_state(inputs)
    actions = dict(path.actions)
    rows: list[dict[str, Any]] = []
    totals = {
        "capital_expenditure_gbp": 0.0,
        "electricity_expenditure_gbp": 0.0,
        "gas_expenditure_gbp": 0.0,
        "maintenance_gbp": 0.0,
        "replacement_cost_gbp": 0.0,
        "stranded_value_gbp": 0.0,
        "residual_value_gbp": 0.0,
        "discounted_lcc_gbp": 0.0,
        "operational_electricity_kgco2e": 0.0,
        "operational_gas_kgco2e": 0.0,
        "upfront_embodied_kgco2e": 0.0,
        "replacement_carbon_kgco2e": 0.0,
        "annual_budget_use_gbp": 0.0,
    }
    replacement_years: list[int] = []
    feasible = True
    for year in range(horizon):
        selected = int(actions.get(year, 0))
        result = transition(
            state,
            selected,
            scenario_run,
            inputs,
            year_exogenous(inputs, scenario_run, year, energy_price_multiplier),
            packages,
            horizon=horizon,
            discount_rate=discount_rate,
        )
        if selected != 0 and not result.feasible:
            feasible = False
            break
        discount = result.financial["discount_factor"]
        for source, target in [
            ("capital_expenditure_gbp", "capital_expenditure_gbp"),
            ("electricity_expenditure_gbp", "electricity_expenditure_gbp"),
            ("gas_expenditure_gbp", "gas_expenditure_gbp"),
            ("maintenance_gbp", "maintenance_gbp"),
            ("replacement_cost_gbp", "replacement_cost_gbp"),
            ("stranded_value_gbp", "stranded_value_gbp"),
            ("residual_value_gbp_credit", "residual_value_gbp"),
        ]:
            totals[target] += result.financial[source] * discount
        totals["discounted_lcc_gbp"] += result.financial["discounted_cashflow_gbp"]
        totals["operational_electricity_kgco2e"] += result.carbon["operational_electricity_kgco2e"]
        totals["operational_gas_kgco2e"] += result.carbon["operational_gas_kgco2e"]
        totals["upfront_embodied_kgco2e"] += result.carbon["upfront_embodied_kgco2e"]
        totals["replacement_carbon_kgco2e"] += result.carbon["replacement_carbon_kgco2e"]
        totals["annual_budget_use_gbp"] += result.financial["capital_expenditure_gbp"]
        if result.diagnostics["scheduled_replacement"]:
            replacement_years.append(year)
        if return_ledger:
            rows.append(
                {
                    "prototype": prototype,
                    "fuel_case": inputs.get("fuel_case", "stock_average"),
                    "scenario": scenario_run["scenario"],
                    "scenario_family": scenario_run["family"],
                    "pathway_id": path.pathway_id,
                    "year": year,
                    "calendar_year": int(config["project"]["appraisal_start_year"]) + year,
                    "current_level_before": state.current_level,
                    "current_level": result.next_state.current_level,
                    "operating_level": result.next_state.current_level,
                    "action_level": selected if selected else np.nan,
                    "selected_action": selected,
                    "applied_action": result.applied_action,
                    "action_mask": "".join("1" if x else "0" for x in result.action_mask),
                    "package_age_before": state.package_age_years,
                    "package_age_after": result.next_state.package_age_years,
                    "layer_1_age_before": state.layer_ages_years[0],
                    "layer_2_age_before": state.layer_ages_years[1],
                    "layer_3_age_before": state.layer_ages_years[2],
                    "layer_1_age_after": result.next_state.layer_ages_years[0],
                    "layer_2_age_after": result.next_state.layer_ages_years[1],
                    "layer_3_age_after": result.next_state.layer_ages_years[2],
                    "layer_1_service_life_years": result.next_state.layer_service_lives_years[0],
                    "layer_2_service_life_years": result.next_state.layer_service_lives_years[1],
                    "layer_3_service_life_years": result.next_state.layer_service_lives_years[2],
                    "electricity_kwh": result.next_state.electricity_kwh_m2yr * inputs["floor_area_m2"],
                    "gas_kwh": result.next_state.gas_kwh_m2yr * inputs["floor_area_m2"],
                    **result.financial,
                    **result.carbon,
                    "vector_reward_cost": result.vector_reward[0],
                    "vector_reward_carbon": result.vector_reward[1],
                    "scalar_reward_balanced": result.scalar_reward,
                    "scheduled_replacement": result.diagnostics["scheduled_replacement"],
                    "terminal_residual_applied": result.diagnostics["terminal_residual_applied"],
                    "event_order_version": result.diagnostics["event_order_version"],
                    "excluded_C1_C4_kgco2e": np.nan,
                    "excluded_D_kgco2e": np.nan,
                    "evidence_basis_codes": "O1;X1;L1;S1",
                }
            )
        state = result.next_state

    if not feasible:
        return {
            "prototype": prototype,
            "fuel_case": inputs.get("fuel_case", "stock_average"),
            "scenario": scenario["scenario"],
            "scenario_family": scenario["family"],
            "pathway_id": path.pathway_id,
            "first_action_year": path.first_action_year,
            "final_level": path.final_level,
            "feasible": False,
        }, None

    ledger = pd.DataFrame(rows) if return_ledger else None
    operational_electricity = totals["operational_electricity_kgco2e"]
    operational_gas = totals["operational_gas_kgco2e"]
    upfront = totals["upfront_embodied_kgco2e"]
    replacement_carbon = totals["replacement_carbon_kgco2e"]
    summary = {
        "prototype": prototype,
        "fuel_case": inputs.get("fuel_case", "stock_average"),
        "scenario": scenario["scenario"],
        "scenario_family": scenario["family"],
        "pathway_id": path.pathway_id,
        "first_action_year": path.first_action_year,
        "final_level": path.final_level,
        "feasible": True,
        "capital_expenditure_gbp": totals["capital_expenditure_gbp"],
        "electricity_expenditure_gbp": totals["electricity_expenditure_gbp"],
        "gas_expenditure_gbp": totals["gas_expenditure_gbp"],
        "maintenance_gbp": totals["maintenance_gbp"],
        "replacement_cost_gbp": totals["replacement_cost_gbp"],
        "stranded_value_gbp": totals["stranded_value_gbp"],
        "residual_value_gbp": totals["residual_value_gbp"],
        # Authoritative totals are the cumulative state returned by the shared
        # transition kernel. Component totals remain separately auditable.
        "discounted_lcc_gbp": state.cumulative_discounted_lcc_gbp,
        "operational_electricity_kgco2e": operational_electricity,
        "operational_gas_kgco2e": operational_gas,
        "operational_carbon_kgco2e": operational_electricity + operational_gas,
        "upfront_embodied_kgco2e": upfront,
        "replacement_carbon_kgco2e": replacement_carbon,
        "total_included_wlc_kgco2e": state.cumulative_physical_carbon_kgco2e,
        "replacement_years": ";".join(map(str, replacement_years)),
        "annual_budget_use_gbp": totals["annual_budget_use_gbp"],
        "wlc_boundary": "planning-level WLC over a 30-year appraisal horizon; included A1-A5/B5 planning envelope, B4, B6; C1-C4 and D excluded",
        "event_order_version": "direct_annual_component_layers_v3",
        "evidence_basis_codes": "O1;X1;L1;S1",
    }
    return summary, ledger if return_ledger else None


def reconcile_ledger(ledger: pd.DataFrame, summary: dict[str, Any]) -> dict[str, float]:
    """Reconcile final objectives against direct annual transition rows."""

    # Explicit left-to-right accumulation reproduces the kernel's annual state
    # update. Python 3.12's built-in sum uses compensated summation and can
    # differ by several 1e-9 for the largest prototype despite identical rows.
    lcc = 0.0
    wlc = 0.0
    for value in ledger["discounted_cashflow_gbp"]:
        lcc += float(value)
    for value in ledger["annual_total_included_wlc_kgco2e"]:
        wlc += float(value)
    return {
        "lcc_difference_gbp": lcc - float(summary["discounted_lcc_gbp"]),
        "wlc_difference_kgco2e": wlc - float(summary["total_included_wlc_kgco2e"]),
    }
