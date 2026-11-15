"""Shared PSDS-informed scenario appraisal functions for revision calculations.

The functions preserve the manuscript's WLC/carbon/ESG and RL/DQN structure.
They convert the authorised PSDS action catalogue and DEC archetype baselines
into repeatable scenario-evaluation inputs. Calculated outcomes are not
observed post-retrofit measurements.
"""

from __future__ import annotations

import hashlib
import itertools
import math
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
from docx import Document


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT
TABLES = OUTPUT / "tables"
FIGURES = OUTPUT / "figures"
MANUSCRIPT = OUTPUT / "Manuscript_1702_result_filled_working_draft.docx"
SEED = 1702
HORIZON = 30
DISCOUNT_RATE = 0.035
ENERGY_VALUE_GBP_KWH = 0.20
CARBON_VALUE_GBP_KG = 0.25
ARCHETYPE_MAP = {
    "Education/school": "Primary school",
    "Civic/office/service": "Office",
    "Health/clinic": "Hospital",
    "Other DEC public-access building": "Other public-access",
}
CORE_ARCHETYPES = ["Primary school", "Office", "Hospital"]
BUDGETS = {"Low": 1.20, "Medium": 2.00, "High": 3.00}


def family_for_action(name: str) -> str:
    lower = name.lower()
    if "heat pump" in lower or "district heating" in lower or "heat network" in lower:
        return "low-carbon heat"
    if "renewable" in lower or "solar" in lower or "battery" in lower:
        return "renewables/storage"
    if "insulation" in lower:
        return "fabric"
    if "lighting" in lower or "hand dryers" in lower:
        return "lighting/equipment"
    if "bem" in lower or "meter" in lower or "controls" in lower or "time switches" in lower:
        return "controls/metering"
    if "ventilation" in lower or "cooling" in lower:
        return "ventilation/cooling"
    if "heating" in lower or "boiler" in lower or "hot water" in lower or "heater" in lower:
        return "heating/hot-water"
    return "enabling/other"


FAMILY_PARAMETERS = {
    "low-carbon heat": (0.44, 0.68, 0.43, 0.38, 0.62),
    "renewables/storage": (0.24, 0.43, 0.28, 0.44, 0.22),
    "fabric": (0.29, 0.27, 0.55, 0.33, 0.56),
    "lighting/equipment": (0.10, 0.10, 0.35, 0.22, 0.12),
    "controls/metering": (0.14, 0.17, 0.35, 0.68, 0.10),
    "ventilation/cooling": (0.12, 0.10, 0.58, 0.28, 0.30),
    "heating/hot-water": (0.22, 0.32, 0.42, 0.30, 0.42),
    "enabling/other": (0.08, 0.08, 0.28, 0.45, 0.18),
}


def load_archetypes() -> pd.DataFrame:
    frame = pd.read_csv(TABLES / "dec_public_proxy_by_archetype.csv")
    frame["archetype_original"] = frame["archetype"]
    frame["archetype"] = frame["archetype"].map(ARCHETYPE_MAP)
    return frame


@lru_cache(maxsize=1)
def archetype_lookup() -> dict[str, dict]:
    return load_archetypes().set_index("archetype").to_dict(orient="index")


def load_weights() -> dict[str, float]:
    frame = pd.read_csv(TABLES / "fahp_top_level_weights_and_consistency.csv")
    mapping = {}
    for row in frame.itertuples():
        name = row.criterion.lower()
        if "economic" in name:
            mapping["economic"] = row.fahp_weight
        elif "environmental" in name:
            mapping["environmental"] = row.fahp_weight
        elif "social" in name:
            mapping["social"] = row.fahp_weight
        elif "governance" in name:
            mapping["governance"] = row.fahp_weight
    return mapping


def build_action_parameters(save: bool = True) -> pd.DataFrame:
    action = pd.read_csv(TABLES / "psds_intervention_action_catalogue_filtered.csv")
    action["action_id"] = [f"A{i:02d}" for i in range(1, len(action) + 1)]
    action["family"] = action["technology_work_type"].map(family_for_action)
    action["spend_per_project_gbp"] = action["confirmed_spend_gbp"] / action["projects_using_measure"]
    median_spend = float(action["spend_per_project_gbp"].median())
    action["spend_imputed"] = action["spend_per_project_gbp"].isna()
    action["spend_per_project_gbp"] = action["spend_per_project_gbp"].fillna(median_spend)
    adoption = np.log1p(action["projects_using_measure"])
    action["adoption_index"] = adoption / adoption.max()
    action["spend_burden"] = np.log1p(action["spend_per_project_gbp"])
    action["spend_burden"] = action["spend_burden"] / action["spend_burden"].median()
    action["budget_units"] = np.clip(0.20 + 0.35 * action["spend_burden"], 0.25, 1.15)
    params = action["family"].map(FAMILY_PARAMETERS)
    action["energy_factor"] = [v[0] for v in params]
    action["carbon_factor"] = [v[1] for v in params]
    action["social_factor"] = [v[2] for v in params]
    action["governance_factor"] = [v[3] for v in params]
    action["disruption_factor"] = [v[4] for v in params]
    evidence_scale = 0.55 + 0.45 * action["adoption_index"]
    for col in ["energy_factor", "carbon_factor", "social_factor", "governance_factor"]:
        action[col] = action[col] * evidence_scale
    weights = load_weights()
    economic = (0.58 * action["energy_factor"] + 0.22 * action["carbon_factor"] + 0.20 * action["adoption_index"])
    action["economic_score"] = economic
    action["environmental_score"] = action["carbon_factor"]
    action["esg_score"] = (
        weights["economic"] * action["economic_score"]
        + weights["environmental"] * action["environmental_score"]
        + weights["social"] * action["social_factor"]
        + weights["governance"] * action["governance_factor"]
    )
    columns = [
        "action_id", "technology_work_type", "family", "projects_using_measure",
        "confirmed_spend_gbp", "spend_per_project_gbp", "spend_imputed",
        "budget_units", "energy_factor", "economic_score", "environmental_score", "social_factor",
        "governance_factor", "disruption_factor", "esg_score",
    ]
    action = action[columns].sort_values("action_id").reset_index(drop=True)
    if save:
        action.to_csv(TABLES / "scenario_action_parameters.csv", index=False)
        pd.DataFrame(
            [
                {"setting": "Horizon (years)", "value": HORIZON, "basis": "Manuscript Table 3"},
                {"setting": "Annual discount rate", "value": DISCOUNT_RATE, "basis": "WLC appraisal setting"},
                {"setting": "Energy valuation (GBP/kWh)", "value": ENERGY_VALUE_GBP_KWH, "basis": "Scenario appraisal setting"},
                {"setting": "Carbon valuation (GBP/kgCO2)", "value": CARBON_VALUE_GBP_KG, "basis": "Scenario appraisal setting"},
                {"setting": "Low budget units", "value": BUDGETS["Low"], "basis": "Proportional scenario envelope"},
                {"setting": "Medium budget units", "value": BUDGETS["Medium"], "basis": "Proportional scenario envelope"},
                {"setting": "High budget units", "value": BUDGETS["High"], "basis": "Proportional scenario envelope"},
            ]
        ).to_csv(TABLES / "scenario_calculation_settings.csv", index=False)
    return action


def build_scenario_trajectories(save: bool = True) -> pd.DataFrame:
    archetypes = load_archetypes()
    actions = build_action_parameters(save=save)
    rows = []
    for arc in archetypes.itertuples():
        for action in actions.itertuples():
            combo = f"{arc.archetype}|{action.action_id}"
            token = int(hashlib.sha256(combo.encode("utf-8")).hexdigest()[:8], 16)
            split_selector = token % 20
            split = "train" if split_selector < 14 else ("validation" if split_selector < 17 else "test")
            for year in range(1, HORIZON + 1):
                ramp = 1.0 - math.exp(-year / 2.0)
                retention = 1.0 - 0.003 * (year - 1)
                energy_reduction = min(0.65, action.energy_factor * ramp * retention)
                carbon_reduction = min(0.78, action.environmental_score * ramp * retention)
                energy = arc.median_energy_intensity_kwh_m2 * (1 - energy_reduction)
                carbon = arc.median_carbon_intensity_kgco2_m2 * (1 - carbon_reduction)
                capex_per_m2 = action.spend_per_project_gbp / arc.median_floor_area_m2
                annuity = DISCOUNT_RATE / (1 - (1 + DISCOUNT_RATE) ** (-HORIZON))
                annualised_cost = (
                    energy * ENERGY_VALUE_GBP_KWH
                    + carbon * CARBON_VALUE_GBP_KG
                    + capex_per_m2 * annuity
                )
                rows.append(
                    {
                        "scenario_id": combo,
                        "archetype": arc.archetype,
                        "action_id": action.action_id,
                        "action_category": action.technology_work_type,
                        "action_family": action.family,
                        "year": year,
                        "split": split,
                        "floor_area_m2": arc.median_floor_area_m2,
                        "baseline_energy_kwh_m2": arc.median_energy_intensity_kwh_m2,
                        "baseline_carbon_kgco2_m2": arc.median_carbon_intensity_kgco2_m2,
                        "budget_units": action.budget_units,
                        "economic_score": action.economic_score,
                        "environmental_score": action.environmental_score,
                        "social_score": action.social_factor,
                        "governance_score": action.governance_factor,
                        "esg_score": action.esg_score,
                        "target_energy_kwh_m2": energy,
                        "target_carbon_kgco2_m2": carbon,
                        "target_annualised_wlc_gbp_m2": annualised_cost,
                    }
                )
    frame = pd.DataFrame(rows)
    if save:
        frame.to_csv(TABLES / "psds_informed_scenario_trajectories.csv", index=False)
        combo = frame.groupby(["scenario_id", "archetype", "action_id", "action_category", "split"], as_index=False).size()
        combo = combo.rename(columns={"size": "trajectory_year_records"})
        combo.to_csv(TABLES / "psds_linked_scenario_validation_split.csv", index=False)
        split_summary = frame.groupby("split", as_index=False).agg(
            scenario_combinations=("scenario_id", "nunique"),
            archetype_action_year_records=("scenario_id", "size"),
        )
        split_summary["unit_of_analysis"] = "archetype-action-year scenario record"
        split_summary.to_csv(TABLES / "psds_linked_scenario_validation_split_summary.csv", index=False)
    return frame


def valid_action_indices(actions: pd.DataFrame, selected: list[int], remaining_budget: float) -> list[int]:
    selected_families = {actions.iloc[i]["family"] for i in selected}
    heat_selected = "low-carbon heat" in selected_families or "heating/hot-water" in selected_families
    valid = []
    for i, row in actions.iterrows():
        if i in selected or row["budget_units"] > remaining_budget + 1e-9:
            continue
        if row["family"] in selected_families:
            continue
        if heat_selected and row["family"] in {"low-carbon heat", "heating/hot-water"}:
            continue
        valid.append(i)
    return valid


def sequence_raw_metrics(archetype: str, selected: list[int], actions: pd.DataFrame, carbon_multiplier: float = 1.0) -> dict:
    if not selected:
        return {"raw_npv": 0.0, "raw_carbon": 0.0, "raw_esg": 0.0, "spend_units": 0.0}
    rows = actions.iloc[selected]
    arc = archetype_lookup()[archetype]
    carbon = float(rows["environmental_score"].sum()) * float(arc["median_carbon_intensity_kgco2_m2"])
    energy_value = float(rows["economic_score"].sum()) * float(arc["median_energy_intensity_kwh_m2"])
    disruption = float(rows["disruption_factor"].sum())
    esg = float(1.0 - np.prod(1.0 - 0.55 * rows["esg_score"].to_numpy(float)))
    carbon_value = carbon * CARBON_VALUE_GBP_KG * carbon_multiplier
    raw_npv = energy_value * ENERGY_VALUE_GBP_KWH + carbon_value - 0.15 * disruption
    return {
        "raw_npv": raw_npv,
        "raw_carbon": carbon,
        "raw_esg": esg,
        "spend_units": float(rows["budget_units"].sum()),
    }


def read_retained_medium_results() -> pd.DataFrame:
    doc = Document(MANUSCRIPT)
    for table in doc.tables:
        header = [cell.text.strip() for cell in table.rows[0].cells]
        if header and header[0] == "Archetype" and "NPV savings" in header[2]:
            rows = []
            for row in table.rows[1:]:
                values = [c.text.strip() for c in row.cells]
                rows.append(
                    {
                        "archetype": values[0],
                        "source_method": values[1],
                        "npv_savings_gbp_m2": float(values[2]),
                        "carbon_reduction_kg_m2": float(values[3]),
                        "esg_score": float(values[4]),
                    }
                )
            result = pd.DataFrame(rows)
            result["method"] = result["source_method"].replace(
                {
                    "Discounted WLC": "WLC-only",
                    "Cost-effectiveness": "GBP/tCO2e ranking",
                    "Static MCDA": "Static MCDA/F-AHP",
                    "Hybrid AI policy": "Hybrid AI / RL policy",
                }
            )
            return result
    raise ValueError("Retained medium-budget comparison table not found in manuscript.")


def draw_bar_chart(path: Path, title: str, labels: list[str], values: list[float], colour=(40, 98, 154)):
    from PIL import Image, ImageDraw, ImageFont

    width, height = 1400, 780
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    draw.text((40, 25), title, fill=(0, 0, 0), font=font)
    left, top, bottom = 90, 90, 650
    max_value = max(values) if values else 1
    bar_w = max(18, int((width - left - 60) / max(1, len(values)) - 10))
    draw.line((left, top, left, bottom), fill=(0, 0, 0), width=2)
    draw.line((left, bottom, width - 30, bottom), fill=(0, 0, 0), width=2)
    for i, (label, value) in enumerate(zip(labels, values)):
        x = left + 15 + i * (bar_w + 10)
        h = int((bottom - top - 30) * value / max_value)
        draw.rectangle((x, bottom - h, x + bar_w, bottom), fill=colour)
        draw.text((x, bottom - h - 18), f"{value:.2f}", fill=(0, 0, 0), font=font)
        draw.text((x, bottom + 10), label[:16], fill=(0, 0, 0), font=font)
    draw.text((left + 5, top - 25), f"Max = {max_value:.2f}", fill=(70, 70, 70), font=font)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def all_compatible_sequences(actions: pd.DataFrame, budget: float, max_actions: int = 5):
    indices = list(range(len(actions)))
    for length in range(1, max_actions + 1):
        for selected in itertools.combinations(indices, length):
            valid = True
            built = []
            remaining = budget
            for idx in selected:
                if idx not in valid_action_indices(actions, built, remaining):
                    valid = False
                    break
                built.append(idx)
                remaining -= float(actions.iloc[idx]["budget_units"])
            if valid:
                yield list(selected)
