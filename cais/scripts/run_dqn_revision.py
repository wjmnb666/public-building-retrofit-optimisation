"""Run the manuscript's RL/DQN constrained retrofit sequencing calculation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from scenario_framework import (
    BUDGETS,
    ROOT,
    SEED,
    TABLES,
    archetype_lookup,
    build_action_parameters,
    load_archetypes,
    sequence_raw_metrics,
    valid_action_indices,
)


REPLAY_BUFFER_SIZE = 4096
TARGET_UPDATE_FREQUENCY = 100
LAMBDA_VIOL = 0.15
GAMMA = 0.95
EPISODES = 1600
LEARNING_RATE = 0.003
BATCH_SIZE = 64
EPSILON_START = 1.0
EPSILON_END = 0.05
EPSILON_DECAY_EPISODES = 1200
MAX_ACTIONS = 5


class QNetwork:
    def __init__(self, state_size, action_size, seed):
        rng = np.random.default_rng(seed)
        self.w1 = rng.normal(0, 0.16, (state_size, 64))
        self.b1 = np.zeros(64)
        self.w2 = rng.normal(0, 0.13, (64, 32))
        self.b2 = np.zeros(32)
        self.w3 = rng.normal(0, 0.10, (32, action_size))
        self.b3 = np.zeros(action_size)

    def predict(self, x):
        x = np.atleast_2d(x)
        h1 = np.maximum(x @ self.w1 + self.b1, 0)
        h2 = np.maximum(h1 @ self.w2 + self.b2, 0)
        return h2 @ self.w3 + self.b3

    def copy_from(self, other):
        for name in ["w1", "b1", "w2", "b2", "w3", "b3"]:
            setattr(self, name, getattr(other, name).copy())

    def update(self, states, actions, targets):
        h1_pre = states @ self.w1 + self.b1
        h1 = np.maximum(h1_pre, 0)
        h2_pre = h1 @ self.w2 + self.b2
        h2 = np.maximum(h2_pre, 0)
        q = h2 @ self.w3 + self.b3
        chosen = q[np.arange(len(states)), actions]
        diff = (chosen - targets) / len(states)
        out_grad = np.zeros_like(q)
        out_grad[np.arange(len(states)), actions] = 2 * diff
        gw3 = h2.T @ out_grad
        gb3 = out_grad.sum(axis=0)
        gh2 = out_grad @ self.w3.T
        gh2[h2_pre <= 0] = 0
        gw2 = h1.T @ gh2
        gb2 = gh2.sum(axis=0)
        gh1 = gh2 @ self.w2.T
        gh1[h1_pre <= 0] = 0
        gw1 = states.T @ gh1
        gb1 = gh1.sum(axis=0)
        for name, grad in [
            ("w1", gw1), ("b1", gb1), ("w2", gw2),
            ("b2", gb2), ("w3", gw3), ("b3", gb3),
        ]:
            setattr(self, name, getattr(self, name) - LEARNING_RATE * np.clip(grad, -5, 5))


class RetrofitEnvironment:
    def __init__(self, actions, archetype, budget_name, carbon_multiplier):
        self.actions = actions
        self.archetype = archetype
        self.budget_name = budget_name
        self.initial_budget = BUDGETS[budget_name]
        self.carbon_multiplier = carbon_multiplier
        self.families = sorted(actions["family"].unique().tolist())
        self.family_codes = actions["family"].map({name: i for i, name in enumerate(self.families)}).to_numpy(int)
        self.budget_costs = actions["budget_units"].to_numpy(float)
        self.economic = actions["economic_score"].to_numpy(float)
        self.environmental = actions["environmental_score"].to_numpy(float)
        self.esg = actions["esg_score"].to_numpy(float)
        self.disruption = actions["disruption_factor"].to_numpy(float)
        arc = archetype_lookup()[archetype]
        self.baseline_energy = float(arc["median_energy_intensity_kwh_m2"])
        self.baseline_carbon = float(arc["median_carbon_intensity_kgco2_m2"])
        self.reset()

    def reset(self):
        self.selected = []
        self.remaining = self.initial_budget
        self.done = False
        return self.state()

    def state(self):
        archetypes = ["Primary school", "Office", "Hospital", "Other public-access"]
        arc = [1.0 if self.archetype == name else 0.0 for name in archetypes]
        metrics = self.metrics()
        selected_families = {self.family_codes[i] for i in self.selected}
        flags = [1.0 if i in selected_families else 0.0 for i in range(len(self.families))]
        return np.asarray(
            arc
            + [self.remaining / self.initial_budget, len(self.selected) / MAX_ACTIONS]
            + [metrics["raw_npv"] / 40, metrics["raw_carbon"] / 200, metrics["raw_esg"]]
            + flags,
            dtype=float,
        )

    def valid(self):
        selected_families = {self.family_codes[i] for i in self.selected}
        heat_codes = {
            self.families.index("low-carbon heat") if "low-carbon heat" in self.families else -1,
            self.families.index("heating/hot-water") if "heating/hot-water" in self.families else -1,
        }
        heat_selected = bool(selected_families.intersection(heat_codes))
        valid = []
        for i in range(len(self.actions)):
            if i in self.selected or self.budget_costs[i] > self.remaining + 1e-9:
                continue
            if self.family_codes[i] in selected_families:
                continue
            if heat_selected and self.family_codes[i] in heat_codes:
                continue
            valid.append(i)
        return valid + [len(self.actions)]  # stop action

    def metrics(self):
        if not self.selected:
            return {"raw_npv": 0.0, "raw_carbon": 0.0, "raw_esg": 0.0, "spend_units": 0.0}
        idx = np.asarray(self.selected, dtype=int)
        carbon = float(self.environmental[idx].sum()) * self.baseline_carbon
        energy_value = float(self.economic[idx].sum()) * self.baseline_energy
        raw_npv = energy_value * 0.20 + carbon * 0.25 * self.carbon_multiplier - 0.15 * float(self.disruption[idx].sum())
        return {
            "raw_npv": raw_npv,
            "raw_carbon": carbon,
            "raw_esg": float(1.0 - np.prod(1.0 - 0.55 * self.esg[idx])),
            "spend_units": float(self.budget_costs[idx].sum()),
        }

    def step(self, action):
        stop_action = len(self.actions)
        if action == stop_action or len(self.selected) >= MAX_ACTIONS:
            self.done = True
            return self.state(), 0.0, True
        if action not in self.valid():
            self.done = True
            return self.state(), -1.0, True
        before = self.metrics()
        self.selected.append(action)
        self.remaining -= self.budget_costs[action]
        after = self.metrics()
        disruption = self.disruption[action]
        reward = (
            0.40 * (after["raw_npv"] - before["raw_npv"]) / 20
            + 0.35 * (after["raw_carbon"] - before["raw_carbon"]) / 100
            + 0.25 * (after["raw_esg"] - before["raw_esg"])
            - LAMBDA_VIOL * disruption
        )
        if len(self.selected) >= MAX_ACTIONS or len(self.valid()) == 1:
            self.done = True
        return self.state(), reward, self.done


def train_network(actions, carbon_multiplier, seed):
    archetypes = load_archetypes()["archetype"].tolist()
    probe = RetrofitEnvironment(actions, archetypes[0], "Medium", carbon_multiplier)
    action_size = len(actions) + 1
    online = QNetwork(len(probe.state()), action_size, seed)
    target = QNetwork(len(probe.state()), action_size, seed + 1)
    target.copy_from(online)
    rng = np.random.default_rng(seed)
    replay = []
    transitions = 0
    for episode in range(EPISODES):
        env = RetrofitEnvironment(
            actions,
            archetypes[int(rng.integers(0, len(archetypes)))],
            list(BUDGETS)[int(rng.integers(0, len(BUDGETS)))],
            carbon_multiplier,
        )
        state = env.reset()
        epsilon = EPSILON_END + (EPSILON_START - EPSILON_END) * max(
            0.0, 1.0 - episode / EPSILON_DECAY_EPISODES
        )
        while not env.done:
            valid = env.valid()
            if rng.random() < epsilon:
                action = int(rng.choice(valid))
            else:
                q = online.predict(state)[0]
                action = max(valid, key=lambda i: q[i])
            next_state, reward, done = env.step(action)
            replay.append((state, action, reward, next_state, done))
            replay = replay[-REPLAY_BUFFER_SIZE:]
            state = next_state
            transitions += 1
            if len(replay) >= BATCH_SIZE:
                indices = rng.choice(len(replay), BATCH_SIZE, replace=False)
                sample = [replay[i] for i in indices]
                states = np.vstack([s[0] for s in sample])
                actions_taken = np.asarray([s[1] for s in sample], dtype=int)
                rewards = np.asarray([s[2] for s in sample])
                following = np.vstack([s[3] for s in sample])
                done_flags = np.asarray([s[4] for s in sample])
                future = target.predict(following).max(axis=1)
                targets = rewards + GAMMA * future * (~done_flags)
                online.update(states, actions_taken, targets)
            if transitions % TARGET_UPDATE_FREQUENCY == 0:
                target.copy_from(online)
    return online


def policy_sequence(network, actions, archetype, budget_name, carbon_multiplier):
    env = RetrofitEnvironment(actions, archetype, budget_name, carbon_multiplier)
    total_reward = 0.0
    while not env.done:
        valid = env.valid()
        q = network.predict(env.state())[0]
        action = max(valid, key=lambda i: q[i])
        _, reward, _ = env.step(action)
        total_reward += reward
    metrics = env.metrics()
    return env.selected, total_reward, metrics


def main():
    actions = build_action_parameters(save=True)
    action_export = actions.copy()
    action_export["hard_mask_rules"] = "Budget exceedance; repeat family; competing low-carbon heat/heating source; repeated action"
    action_export["soft_penalty"] = "Disruption factor multiplied by lambda_viol"
    action_export.to_csv(TABLES / "dqn_action_catalogue.csv", index=False)
    rows = []
    for carbon_label, multiplier, seed in [("Baseline carbon price", 1.0, SEED), ("2x carbon price", 2.0, SEED + 200)]:
        network = train_network(actions, multiplier, seed)
        np.savez(
            TABLES / ("dqn_revision_weights_baseline.npz" if multiplier == 1 else "dqn_revision_weights_2x_carbon.npz"),
            w1=network.w1, b1=network.b1, w2=network.w2, b2=network.b2, w3=network.w3, b3=network.b3,
        )
        for archetype in load_archetypes()["archetype"]:
            for budget_name in BUDGETS:
                selected, reward, metrics = policy_sequence(network, actions, archetype, budget_name, multiplier)
                selected_rows = actions.iloc[selected] if selected else actions.iloc[[]]
                rows.append(
                    {
                        "carbon_price_scenario": carbon_label,
                        "archetype": archetype,
                        "budget_level": budget_name,
                        "budget_units": BUDGETS[budget_name],
                        "selected_action_ids": "; ".join(selected_rows["action_id"].tolist()) or "No action",
                        "selected_sequence": "; ".join(selected_rows["technology_work_type"].tolist()) or "No action",
                        "used_budget_units": metrics["spend_units"],
                        "policy_reward": reward,
                        **metrics,
                    }
                )
    pd.DataFrame(rows).to_csv(TABLES / "dqn_selected_sequences.csv", index=False)
    config = [
        ("Action-space size", f"{len(actions) + 1} (37 intervention actions plus stop)"),
        ("State variables", "Archetype; remaining budget; step; cumulative NPV/carbon/ESG indices; selected-family flags"),
        ("Budget envelopes", "Low 1.20; Medium 2.00; High 3.00 normalised spend-burden units"),
        ("Hard feasibility mask", "Budget; repeat action/family; competing heating source; five-action terminal horizon"),
        ("Penalty allocation", "Feasible disruption burden only"),
        ("lambda_viol", str(LAMBDA_VIOL)),
        ("Replay-buffer size", str(REPLAY_BUFFER_SIZE)),
        ("Target-network update frequency", f"{TARGET_UPDATE_FREQUENCY} transitions"),
        ("Exploration schedule", f"epsilon {EPSILON_START:.2f} to {EPSILON_END:.2f} over {EPSILON_DECAY_EPISODES} episodes"),
        ("Discount factor gamma", str(GAMMA)),
        ("Training episodes", str(EPISODES)),
        ("Mini-batch size", str(BATCH_SIZE)),
        ("Random seed", str(SEED)),
    ]
    pd.DataFrame(config, columns=["configuration_item", "value"]).to_csv(TABLES / "dqn_revision_config.csv", index=False)
    print(pd.DataFrame(config, columns=["configuration_item", "value"]).to_string(index=False))
    print(pd.DataFrame(rows).head(12).to_string(index=False))


if __name__ == "__main__":
    main()
