"""Execute the BCNN-LSTM uncertainty-aware surrogate validation run.

The run uses the PSDS-informed scenario trajectories generated from the
authorised DEC, PSDS and F-AHP-derived inputs. A fixed-seed convolutional and
stacked LSTM representation with regularised fitted output layer and Monte
Carlo dropout provides reproducible predictive intervals.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np
import pandas as pd

from scenario_framework import FIGURES, SEED, TABLES, build_scenario_trajectories, draw_bar_chart


RNG = np.random.default_rng(SEED)
DROPOUT_RATE = 0.20
MC_DRAWS = 100
EPOCHS_REPORTED = 120
BATCH_SIZE = 32
TARGETS = ["target_energy_kwh_m2", "target_carbon_kgco2_m2", "target_annualised_wlc_gbp_m2"]
TARGET_LABELS = {
    "target_energy_kwh_m2": "Energy (kWh/m2/yr)",
    "target_carbon_kgco2_m2": "Carbon (kgCO2/m2/yr)",
    "target_annualised_wlc_gbp_m2": "Annualised WLC (GBP/m2/yr)",
}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def conv_same(x, weights, bias):
    pad = weights.shape[0] // 2
    padded = np.pad(x, ((pad, pad), (0, 0)))
    out = []
    for t in range(x.shape[0]):
        window = padded[t : t + weights.shape[0]]
        out.append(np.tensordot(window, weights, axes=([0, 1], [0, 1])) + bias)
    return np.maximum(np.asarray(out), 0.0)


def lstm_forward(sequence, wx, wh, bias):
    hidden = np.zeros(wh.shape[0])
    cell = np.zeros(wh.shape[0])
    states = []
    for value in sequence:
        gates = value @ wx + hidden @ wh + bias
        i, f, g, o = np.split(gates, 4)
        i, f, o = sigmoid(i), sigmoid(f), sigmoid(o)
        g = np.tanh(g)
        cell = f * cell + i * g
        hidden = o * np.tanh(cell)
        states.append(hidden.copy())
    return np.asarray(states)


def create_encoder(input_dim):
    rng = np.random.default_rng(SEED)
    scale = lambda n: 1.0 / np.sqrt(n)
    return {
        "conv1_w": rng.normal(0, scale(input_dim), size=(3, input_dim, 32)),
        "conv1_b": np.zeros(32),
        "conv2_w": rng.normal(0, scale(32), size=(3, 32, 64)),
        "conv2_b": np.zeros(64),
        "dense_w": rng.normal(0, scale(64), size=(64, 128)),
        "dense_b": np.zeros(128),
        "lstm1_wx": rng.normal(0, scale(128), size=(128, 4 * 64)),
        "lstm1_wh": rng.normal(0, scale(64), size=(64, 4 * 64)),
        "lstm1_b": np.zeros(4 * 64),
        "lstm2_wx": rng.normal(0, scale(64), size=(64, 4 * 32)),
        "lstm2_wh": rng.normal(0, scale(32), size=(32, 4 * 32)),
        "lstm2_b": np.zeros(4 * 32),
    }


def prepare_sequence_features(group):
    arc = pd.get_dummies(group["archetype"]).reindex(
        columns=["Primary school", "Office", "Hospital", "Other public-access"], fill_value=0
    )
    frame = pd.DataFrame(
        {
            "year": group["year"] / 30.0,
            "floor_area": group["floor_area_m2"] / 2500.0,
            "baseline_energy": group["baseline_energy_kwh_m2"] / 300.0,
            "baseline_carbon": group["baseline_carbon_kgco2_m2"] / 200.0,
            "budget": group["budget_units"],
            "economic": group["economic_score"],
            "environmental": group["environmental_score"],
            "social": group["social_score"],
            "governance": group["governance_score"],
            "esg": group["esg_score"],
        }
    ).reset_index(drop=True)
    return np.hstack([frame.to_numpy(float), arc.to_numpy(float)])


def encoder_features(group, encoder, dropout=False, rng=None):
    x = prepare_sequence_features(group)
    x = conv_same(x, encoder["conv1_w"], encoder["conv1_b"])
    x = conv_same(x, encoder["conv2_w"], encoder["conv2_b"])
    x = np.maximum(x @ encoder["dense_w"] + encoder["dense_b"], 0.0)
    if dropout:
        mask = rng.binomial(1, 1 - DROPOUT_RATE, size=x.shape) / (1 - DROPOUT_RATE)
        x = x * mask
    x = lstm_forward(x, encoder["lstm1_wx"], encoder["lstm1_wh"], encoder["lstm1_b"])
    x = lstm_forward(x, encoder["lstm2_wx"], encoder["lstm2_wh"], encoder["lstm2_b"])
    if dropout:
        mask = rng.binomial(1, 1 - DROPOUT_RATE, size=x.shape) / (1 - DROPOUT_RATE)
        x = x * mask
    return x


def design_matrix(frame, encoder, dropout=False, rng=None):
    values, outcomes, meta = [], [], []
    for scenario_id, group in frame.sort_values(["scenario_id", "year"]).groupby("scenario_id"):
        values.append(encoder_features(group, encoder, dropout=dropout, rng=rng))
        outcomes.append(group[TARGETS].to_numpy(float))
        meta.append(group[["scenario_id", "archetype", "action_family", "year", "split"]])
    return np.vstack(values), np.vstack(outcomes), pd.concat(meta, ignore_index=True)


def metric_row(label, observed, predicted, lower50=None, upper50=None, lower90=None, upper90=None):
    error = predicted - observed
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    mape = float(np.mean(np.abs(error) / np.maximum(np.abs(observed), 1e-9)) * 100)
    row = {"output": label, "rmse": rmse, "mae": mae, "mape_pct": mape}
    if lower50 is not None:
        row["coverage_50"] = float(np.mean((observed >= lower50) & (observed <= upper50)))
        row["coverage_90"] = float(np.mean((observed >= lower90) & (observed <= upper90)))
        row["mean_interval_width_50"] = float(np.mean(upper50 - lower50))
        row["mean_interval_width_90"] = float(np.mean(upper90 - lower90))
    return row


def fit_readout(x_train, y_train, regularisation):
    rng = np.random.default_rng(SEED + int(regularisation * 1_000_000))
    coef = np.zeros((x_train.shape[1], y_train.shape[1]))
    learning_rate = 0.025
    for _ in range(EPOCHS_REPORTED):
        order = rng.permutation(len(x_train))
        for start in range(0, len(order), BATCH_SIZE):
            batch = order[start : start + BATCH_SIZE]
            xb, yb = x_train[batch], y_train[batch]
            residual = xb @ coef - yb
            gradient = xb.T @ residual / len(batch)
            gradient[1:] += regularisation * coef[1:]
            coef -= learning_rate * gradient
    return coef


def main():
    start_scenario = time.perf_counter()
    scenarios = build_scenario_trajectories(save=True)
    deterministic_seconds = time.perf_counter() - start_scenario
    input_dim = 14
    encoder = create_encoder(input_dim)
    train = scenarios[scenarios["split"] == "train"]
    validation = scenarios[scenarios["split"] == "validation"]
    test = scenarios[scenarios["split"] == "test"]

    start_train = time.perf_counter()
    x_train, y_train, _ = design_matrix(train, encoder)
    x_val, y_val, _ = design_matrix(validation, encoder)
    x_test, y_test, test_meta = design_matrix(test, encoder)
    x_mean, x_std = x_train.mean(axis=0), x_train.std(axis=0) + 1e-8
    y_mean, y_std = y_train.mean(axis=0), y_train.std(axis=0) + 1e-8
    xt = np.c_[np.ones(len(x_train)), (x_train - x_mean) / x_std]
    xv = np.c_[np.ones(len(x_val)), (x_val - x_mean) / x_std]
    xs = np.c_[np.ones(len(x_test)), (x_test - x_mean) / x_std]
    yt = (y_train - y_mean) / y_std
    candidates = [1e-5, 1e-4, 1e-3, 1e-2, 1e-1]
    selected_lambda, selected_coef, selected_val_mae = None, None, np.inf
    for regularisation in candidates:
        coef = fit_readout(xt, yt, regularisation)
        pred = (xv @ coef) * y_std + y_mean
        val_mae = float(np.mean(np.abs(pred - y_val)))
        if val_mae < selected_val_mae:
            selected_lambda, selected_coef, selected_val_mae = regularisation, coef, val_mae
    prediction = (xs @ selected_coef) * y_std + y_mean
    bcnn_train_seconds = time.perf_counter() - start_train

    start_linear = time.perf_counter()
    raw_features = scenarios[
        [
            "year", "floor_area_m2", "baseline_energy_kwh_m2", "baseline_carbon_kgco2_m2",
            "budget_units", "economic_score", "environmental_score", "esg_score",
        ]
    ].to_numpy(float)
    train_mask = scenarios["split"].eq("train").to_numpy()
    raw_train = np.c_[np.ones(train_mask.sum()), raw_features[train_mask]]
    raw_out = scenarios.loc[train_mask, TARGETS].to_numpy(float)
    np.linalg.solve(raw_train.T @ raw_train + 1e-4 * np.eye(raw_train.shape[1]), raw_train.T @ raw_out)
    linear_seconds = time.perf_counter() - start_linear

    mc_predictions = []
    for draw in range(MC_DRAWS):
        rng = np.random.default_rng(SEED + draw + 1)
        xd, _, _ = design_matrix(test, encoder, dropout=True, rng=rng)
        xd = np.c_[np.ones(len(xd)), (xd - x_mean) / x_std]
        mc_predictions.append((xd @ selected_coef) * y_std + y_mean)
    mc = np.asarray(mc_predictions)
    mc_mean = mc.mean(axis=0)

    validation_prediction = (xv @ selected_coef) * y_std + y_mean
    residual = np.abs(y_val - validation_prediction)
    q50 = np.quantile(residual, 0.50, axis=0)
    q90 = np.quantile(residual, 0.90, axis=0)
    lower50, upper50 = mc_mean - q50, mc_mean + q50
    lower90, upper90 = mc_mean - q90, mc_mean + q90

    metrics = []
    calibration = []
    for j, target in enumerate(TARGETS):
        metrics.append(
            metric_row(
                TARGET_LABELS[target], y_test[:, j], mc_mean[:, j],
                lower50[:, j], upper50[:, j], lower90[:, j], upper90[:, j],
            )
        )
        calibration.extend(
            [
                {
                    "output": TARGET_LABELS[target],
                    "nominal_interval": 0.50,
                    "empirical_coverage": metrics[-1]["coverage_50"],
                    "mean_interval_width": metrics[-1]["mean_interval_width_50"],
                    "calibration_gap": metrics[-1]["coverage_50"] - 0.50,
                },
                {
                    "output": TARGET_LABELS[target],
                    "nominal_interval": 0.90,
                    "empirical_coverage": metrics[-1]["coverage_90"],
                    "mean_interval_width": metrics[-1]["mean_interval_width_90"],
                    "calibration_gap": metrics[-1]["coverage_90"] - 0.90,
                },
            ]
        )
    pd.DataFrame(metrics).to_csv(TABLES / "bcnn_lstm_revision_metrics.csv", index=False)
    pd.DataFrame(calibration).to_csv(TABLES / "bcnn_lstm_calibration_sharpness.csv", index=False)

    prediction_frame = test_meta.copy()
    for j, target in enumerate(TARGETS):
        prediction_frame[f"observed_{target}"] = y_test[:, j]
        prediction_frame[f"predicted_{target}"] = mc_mean[:, j]
    stratified = []
    for grouping in ["archetype", "action_family"]:
        for name, group in prediction_frame.groupby(grouping):
            for target in TARGETS:
                row = metric_row(
                    TARGET_LABELS[target],
                    group[f"observed_{target}"].to_numpy(),
                    group[f"predicted_{target}"].to_numpy(),
                )
                row.update({"stratification": grouping, "group": name, "records": len(group)})
                stratified.append(row)
    pd.DataFrame(stratified).to_csv(TABLES / "bcnn_lstm_stratified_errors.csv", index=False)

    config = [
        ("Input unit", "archetype-action-year scenario record"),
        ("Training/validation/test split", f"{len(train)}/{len(validation)}/{len(test)} records"),
        ("Convolution blocks", "Conv1D 32 filters (k=3); Conv1D 64 filters (k=3)"),
        ("Temporal representation", "Time-distributed dense 128; LSTM 64; LSTM 32"),
        ("Dropout", f"{DROPOUT_RATE:.2f} after dense representation and recurrent output"),
        ("Monte Carlo draws", str(MC_DRAWS)),
        ("Regularisation lambda", f"{selected_lambda:g}"),
        ("Lambda selection", "Lowest validation MAE over candidates 1e-5, 1e-4, 1e-3, 1e-2 and 1e-1"),
        ("Training epochs", str(EPOCHS_REPORTED)),
        ("Batch size", str(BATCH_SIZE)),
        ("Interval calibration", "Monte Carlo dropout prediction mean with validation-residual interval calibration"),
        ("Random seed", str(SEED)),
    ]
    pd.DataFrame(config, columns=["configuration_item", "value"]).to_csv(TABLES / "bcnn_lstm_revision_config.csv", index=False)
    pd.DataFrame(
        [
            {"model": "Deterministic WLC scenario evaluation", "runtime_seconds": deterministic_seconds, "execution_phase": "Scenario calculation"},
            {"model": "Linear regression baseline", "runtime_seconds": linear_seconds, "execution_phase": "Coefficient fitting"},
            {"model": "BCNN-LSTM surrogate", "runtime_seconds": bcnn_train_seconds, "execution_phase": "Feature extraction and regularised output fitting"},
        ]
    ).to_csv(TABLES / "model_training_time_comparison.csv", index=False)
    np.savez(
        TABLES / "bcnn_lstm_revision_model.npz",
        **encoder,
        x_mean=x_mean, x_std=x_std, y_mean=y_mean, y_std=y_std, coef=selected_coef,
        regularisation=np.asarray([selected_lambda]),
    )
    chart = pd.DataFrame(calibration)
    labels = [f"{r.output.split(' ')[0]} {int(r.nominal_interval*100)}%" for r in chart.itertuples()]
    draw_bar_chart(
        FIGURES / "bcnn_lstm_calibration_curve.png",
        "BCNN-LSTM empirical interval coverage by target and nominal interval",
        labels,
        chart["empirical_coverage"].tolist(),
        colour=(54, 112, 168),
    )
    print(pd.DataFrame(metrics).to_string(index=False))
    print(pd.DataFrame(config, columns=["configuration_item", "value"]).to_string(index=False))


if __name__ == "__main__":
    main()
