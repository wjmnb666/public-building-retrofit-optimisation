"""Exact enumeration over the shared direct annual transition kernel."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import OUTPUTS
from .data_sources import P2, PROTOTYPES, load_config
from .model import Pathway, enumerate_pathways, evaluate_path, reconcile_ledger, scenario_registry


PREFERENCES = {
    "cost_focused": (0.8, 0.2),
    "balanced": (0.5, 0.5),
    "carbon_focused": (0.2, 0.8),
}


def pareto_mask(cost: np.ndarray, carbon: np.ndarray) -> np.ndarray:
    """Exact nondominated mask for two minimisation objectives."""

    order = np.lexsort((carbon, cost))
    keep = np.zeros(len(cost), dtype=bool)
    best_carbon = np.inf
    for index in order:
        if carbon[index] < best_carbon - 1e-10:
            keep[index] = True
            best_carbon = carbon[index]
    return keep


def normalise_objectives(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for source, target in [
        ("discounted_lcc_gbp", "normalised_lcc"),
        ("total_included_wlc_kgco2e", "normalised_wlc"),
    ]:
        low = float(result[source].min())
        high = float(result[source].max())
        result[target] = 0.0 if np.isclose(high, low) else (result[source] - low) / (high - low)
        result[f"{target}_min"] = low
        result[f"{target}_max"] = high
        result[f"{target}_scale"] = high - low
    return result


def _evaluate_group(
    prototype: str,
    inputs: dict[str, Any],
    scenario: dict[str, Any],
    paths: list[Pathway],
) -> tuple[pd.DataFrame, int]:
    summaries: list[dict[str, Any]] = []
    infeasible = 0
    for path in paths:
        summary, _ = evaluate_path(prototype, inputs, path, scenario)
        if summary["feasible"]:
            summaries.append(summary)
        else:
            infeasible += 1
    if not summaries:
        raise RuntimeError(f"No feasible direct-transition pathways for {prototype}/{scenario['scenario']}")
    return normalise_objectives(pd.DataFrame(summaries)), infeasible


def _select_preferred(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for preference, (cost_weight, carbon_weight) in PREFERENCES.items():
        score = cost_weight * frame["normalised_lcc"] + carbon_weight * frame["normalised_wlc"]
        index = int(score.idxmin())
        selected = frame.loc[index].to_dict()
        selected.update(
            {
                "preference": preference,
                "cost_weight": cost_weight,
                "carbon_weight": carbon_weight,
                "scalarised_score": float(score.loc[index]),
                "action_timeline": selected["pathway_id"],
                "feasible_path_count": int(len(frame)),
                "exact_pareto_count": int(frame["is_exact_pareto"].sum()),
            }
        )
        rows.append(selected)
    return rows


def _p2_heating_consistent_inputs(stock: dict[str, Any]) -> dict[str, Any]:
    result = dict(stock)
    start = int(load_config()["project"]["appraisal_start_year"])
    factor = float(stock["grid_average"][start])
    result["gas_kwh_m2yr"] = 0.0
    result["electricity_kwh_m2yr"] = float(stock["epc_operational_emissions_kgco2e_m2yr"]) / factor
    result["fuel_case"] = "heating_consistent_electricity_only"
    return result


def _run_p2_fuel_split(
    prototype_data: dict[str, dict[str, Any]], paths: list[Pathway]
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    central = next(item for item in scenario_registry() if item["scenario"] == "central")
    stock = dict(prototype_data[P2])
    stock["fuel_case"] = "stock_average"
    heating = _p2_heating_consistent_inputs(stock)
    input_rows = [
        {
            "prototype": P2,
            "fuel_case": "stock_average",
            "electricity_kwh_m2yr": stock["electricity_kwh_m2yr"],
            "gas_kwh_m2yr": stock["gas_kwh_m2yr"],
            "derivation": "retained ND-NEED-derived calibrated electricity and gas intensities",
            "interpretation": "medium offices with grid electricity as the recorded main heating fuel; not purely electric",
        },
        {
            "prototype": P2,
            "fuel_case": "heating_consistent_electricity_only",
            "electricity_kwh_m2yr": heating["electricity_kwh_m2yr"],
            "gas_kwh_m2yr": 0.0,
            "derivation": "valid prototype EPC operational-emissions mean divided by declared starting grid-average electricity carbon factor",
            "interpretation": "prospective heating-consistent sensitivity; does not overwrite the stock-average baseline",
        },
    ]
    frames: dict[str, pd.DataFrame] = {}
    preferred_rows: list[dict[str, Any]] = []
    for name, inputs in [("stock_average", stock), ("heating_consistent_electricity_only", heating)]:
        frame, _ = _evaluate_group(P2, inputs, central, paths)
        mask = pareto_mask(
            frame["discounted_lcc_gbp"].to_numpy(float),
            frame["total_included_wlc_kgco2e"].to_numpy(float),
        )
        frame["is_exact_pareto"] = mask
        frames[name] = frame
        preferred_rows.extend(_select_preferred(frame))
    preferred = pd.DataFrame(preferred_rows)
    pivot = preferred.pivot(index="preference", columns="fuel_case", values="pathway_id")
    stability_rows = []
    for preference in PREFERENCES:
        stock_row = preferred[
            (preferred["preference"] == preference) & (preferred["fuel_case"] == "stock_average")
        ].iloc[0]
        heat_row = preferred[
            (preferred["preference"] == preference)
            & (preferred["fuel_case"] == "heating_consistent_electricity_only")
        ].iloc[0]
        stability_rows.append(
            {
                "prototype": P2,
                "preference": preference,
                "stock_average_pathway": stock_row["pathway_id"],
                "heating_consistent_pathway": heat_row["pathway_id"],
                "pathway_stable": stock_row["pathway_id"] == heat_row["pathway_id"],
                "first_action_year_difference": (
                    np.nan
                    if pd.isna(stock_row["first_action_year"]) or pd.isna(heat_row["first_action_year"])
                    else int(heat_row["first_action_year"] - stock_row["first_action_year"])
                ),
                "lcc_difference_gbp": heat_row["discounted_lcc_gbp"] - stock_row["discounted_lcc_gbp"],
                "operational_carbon_difference_kgco2e": heat_row["operational_carbon_kgco2e"]
                - stock_row["operational_carbon_kgco2e"],
                "embodied_carbon_difference_kgco2e": (
                    heat_row["upfront_embodied_kgco2e"]
                    + heat_row["replacement_carbon_kgco2e"]
                    - stock_row["upfront_embodied_kgco2e"]
                    - stock_row["replacement_carbon_kgco2e"]
                ),
                "total_wlc_difference_kgco2e": heat_row["total_included_wlc_kgco2e"]
                - stock_row["total_included_wlc_kgco2e"],
                "conclusion": (
                    "fuel-allocation-sensitive"
                    if stock_row["pathway_id"] != heat_row["pathway_id"]
                    else "robustness finding: preferred pathway unchanged"
                ),
            }
        )
    inputs_frame = pd.DataFrame(input_rows)
    stability = pd.DataFrame(stability_rows)
    inputs_frame.to_csv(OUTPUTS / "p2_fuel_split_inputs.csv", index=False)
    preferred.to_csv(OUTPUTS / "p2_fuel_split_exact.csv", index=False)
    stability.to_csv(OUTPUTS / "p2_fuel_split_rank_stability.csv", index=False)
    return preferred, stability, frames


def run_exact(prototype_data: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Enumerate all registered direct-transition scenarios and export ledgers."""

    paths = enumerate_pathways(int(load_config()["project"]["horizon_years"]))
    scenarios = scenario_registry()
    path_lookup = {path.pathway_id: path for path in paths}
    scenario_lookup = {item["scenario"]: item for item in scenarios}
    all_frames: list[pd.DataFrame] = []
    pareto_frames: list[pd.DataFrame] = []
    preferred_rows: list[dict[str, Any]] = []
    count_rows: list[dict[str, Any]] = []

    for scenario in scenarios:
        for prototype in PROTOTYPES:
            frame, infeasible = _evaluate_group(prototype, prototype_data[prototype], scenario, paths)
            mask = pareto_mask(
                frame["discounted_lcc_gbp"].to_numpy(float),
                frame["total_included_wlc_kgco2e"].to_numpy(float),
            )
            frame["is_exact_pareto"] = mask
            frame["is_dominated"] = ~mask
            all_frames.append(frame)
            pareto_frames.append(frame[mask].copy())
            preferred_rows.extend(_select_preferred(frame))
            count_rows.append(
                {
                    "scenario": scenario["scenario"],
                    "prototype": prototype,
                    "total_candidate_pathways": len(paths),
                    "feasible_pathways": len(frame),
                    "infeasible_pathways": infeasible,
                    "dominated_pathways": int((~mask).sum()),
                    "pareto_front_size": int(mask.sum()),
                }
            )

    exact_all = pd.concat(all_frames, ignore_index=True)
    exact_pareto = pd.concat(pareto_frames, ignore_index=True)
    preferred = pd.DataFrame(preferred_rows)
    counts = pd.DataFrame(count_rows)
    lcc_ledgers: list[pd.DataFrame] = []
    wlc_ledgers: list[pd.DataFrame] = []
    components: list[dict[str, Any]] = []
    reconciliations: list[dict[str, Any]] = []
    for row in preferred.to_dict(orient="records"):
        scenario = scenario_lookup[str(row["scenario"])]
        summary, ledger = evaluate_path(
            str(row["prototype"]),
            prototype_data[str(row["prototype"])],
            path_lookup[str(row["pathway_id"])],
            scenario,
            return_ledger=True,
        )
        assert ledger is not None
        ledger.insert(4, "preference", row["preference"])
        lcc_ledgers.append(ledger.copy())
        wlc_ledgers.append(ledger.copy())
        reconciliations.append(
            {
                "prototype": row["prototype"],
                "scenario": row["scenario"],
                "preference": row["preference"],
                "pathway_id": row["pathway_id"],
                **reconcile_ledger(ledger, summary),
            }
        )
        components.append(
            {
                "prototype": row["prototype"],
                "scenario": row["scenario"],
                "preference": row["preference"],
                "pathway_id": row["pathway_id"],
                "operational_electricity_kgco2e": summary["operational_electricity_kgco2e"],
                "operational_gas_kgco2e": summary["operational_gas_kgco2e"],
                "operational_carbon_kgco2e": summary["operational_carbon_kgco2e"],
                "upfront_A1_A5_or_B5_envelope_kgco2e": summary["upfront_embodied_kgco2e"],
                "B4_replacement_carbon_kgco2e": summary["replacement_carbon_kgco2e"],
                "total_included_WLC_kgco2e": summary["total_included_wlc_kgco2e"],
                "excluded_C1_C4": "excluded_not_quantified",
                "excluded_D": "excluded_not_quantified",
                "terminology": "planning-level WLC over a 30-year appraisal horizon",
            }
        )

    lcc_detail = pd.concat(lcc_ledgers, ignore_index=True)
    wlc_detail = pd.concat(wlc_ledgers, ignore_index=True)
    reconciliation = pd.DataFrame(reconciliations)
    exact_all.to_csv(OUTPUTS / "exact_all_pathways.csv", index=False)
    exact_pareto.to_csv(OUTPUTS / "exact_pareto_frontier.csv", index=False)
    preferred.to_csv(OUTPUTS / "exact_preferred_paths.csv", index=False)
    preferred.to_csv(OUTPUTS / "dynamic_scenario_results.csv", index=False)
    counts.to_csv(OUTPUTS / "exact_pathway_counts.csv", index=False)
    lcc_detail.to_csv(OUTPUTS / "lcc_annual_detail.csv", index=False)
    wlc_detail.to_csv(OUTPUTS / "wlc_annual_detail.csv", index=False)
    pd.DataFrame(components).to_csv(OUTPUTS / "wlc_component_summary.csv", index=False)
    reconciliation.to_csv(OUTPUTS / "ledger_reconciliation.csv", index=False)

    central = preferred[preferred["scenario"] == "central"]
    diagnosis_rows = []
    for prototype in PROTOTYPES:
        group = central[central["prototype"] == prototype]
        action_years = group["first_action_year"].dropna().astype(int)
        diagnosis_rows.append(
            {
                "prototype": prototype,
                "central_preference_count": len(group),
                "year0_preferred_count": int((action_years == 0).sum()),
                "no_action_preferred_count": int(group["first_action_year"].isna().sum()),
                "all_action_paths_start_year0": bool(len(action_years) and (action_years == 0).all()),
                "diagnosis": "Direct same-year savings and operational-carbon reductions favour early action when the annual budget is non-binding; capex discounting, embodied carbon and later replacements can offset but do not create an imposed Year-0 rule.",
            }
        )
    year0 = pd.DataFrame(diagnosis_rows)
    year0.to_csv(OUTPUTS / "year0_action_diagnosis.csv", index=False)
    p2_preferred, p2_stability, p2_frames = _run_p2_fuel_split(prototype_data, paths)

    result = {
        "path_count_unconstrained": len(paths),
        "scenario_count": len(scenarios),
        "exact_rows": len(exact_all),
        "pareto_rows": len(exact_pareto),
        "preferred_rows": len(preferred),
        "ledgers_reconcile": bool(
            reconciliation["lcc_difference_gbp"].abs().max() < 1e-9
            and reconciliation["wlc_difference_kgco2e"].abs().max() < 1e-9
        ),
        "central_preferred": central[
            ["prototype", "preference", "pathway_id", "discounted_lcc_gbp", "total_included_wlc_kgco2e"]
        ].to_dict(orient="records"),
        "p2_fuel_split_pathway_stability_rate": float(p2_stability["pathway_stable"].mean()),
    }
    return {
        "summary": result,
        "all": exact_all,
        "pareto": exact_pareto,
        "preferred": preferred,
        "year0": year0,
        "p2_preferred": p2_preferred,
        "p2_frames": p2_frames,
    }

