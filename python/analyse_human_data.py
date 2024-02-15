from itertools import product
import json
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.modelling import expected_acc_approx

NUM_ROUNDS = 3
NUM_CHOICES = 10

sns.set_theme()
sns.set_context("paper")


def load_sessions(cursor):
    res = cur.execute("SELECT id,experiment_id,user_id,cond FROM sessions")
    sessions = res.fetchall()
    return {
        s[0]: {"id": s[0], "experiment_id": s[1], "user_id": s[2], "condition": s[3]}
        for s in sessions
    }


def load_choices(cursor):
    res = cur.execute(
        "SELECT id,session_id,round,patch_size,row1,col1,row2,col2,selected FROM choices"
    )
    choices, choice_dict = res.fetchall(), {}
    for c in choices:
        if c[1] not in choice_dict:
            choice_dict[c[1]] = []
        choice_dict[c[1]].append(
            {
                "id": c[0],
                "session_id": c[1],
                "round": c[2],
                "patch_size": c[3],
                "tile_1": (c[4], c[5]),
                "tile_2": (c[6], c[7]),
                "selected": c[8],
            }
        )
    return choice_dict


def compute_accuracy_values(session_dict, choice_dict):
    heatmaps, acc_data = {}, []

    # load each round's heatmap from text file
    for round in range(NUM_ROUNDS):
        with open(f"../heatmaps/{round + 1}/1.txt", "r") as f:
            heatmaps[round + 1] = np.array(json.load(f))

    for sid, choices in choice_dict.items():
        accs, pss = {}, {}
        for c in choices:
            vals = np.array([heatmaps[c["round"]][c[f"tile_{i}"]] for i in (1, 2)])
            correct = vals[c["selected"]] >= vals[1 - c["selected"]]
            accs[c["round"]] = accs.get(c["round"], 0) + correct

            if c["round"] not in pss:
                pss[c["round"]] = c["patch_size"]

        for round in accs:
            acc_data.append(
                {
                    "session_id": sid,
                    "condition": session_dict[sid]["condition"],
                    "round": round,
                    "patch_size": pss[round] ** 2,
                    "accuracy": accs[round] / NUM_CHOICES,
                }
            )

    return pd.DataFrame(acc_data)


# # load sessions and choices
# conn = sqlite3.connect("../store.db")
# cur = conn.cursor()

# sessions = load_sessions(cur)
# choices = load_choices(cur)

# conn.close()

# # compute accuracy values
# acc_data = compute_accuracy_values(sessions, choices)

# # make plot of accuracy vs patch size
# fig, ax = plt.subplots()
# sns.lineplot(data=acc_data, x="patch_size", y="accuracy", ax=ax)
# plt.show()

r = 5
n = 20
costs = {
    1: 100,
    2: 40,
    4: 10,
    8: 1,
}
c_vals = [0.1, 1, 5]
data, patch_lengths = [], [1, 2, 4, 8]

for texture in ["rough", "smooth"]:
    for round in range(10):
        hmaps, e_accs = {}, {}
        for pl in patch_lengths:
            with open(f"../heatmaps/{texture}/{round + 1}/{pl}.txt", "r") as f:
                hmap = np.array(json.load(f))
                hmaps[pl] = np.repeat(np.repeat(hmap, pl, axis=0), pl, axis=1)
                e_accs[pl] = expected_acc_approx(hmaps[1], hmaps[pl], 0.001)

        for pl, c in product(patch_lengths, c_vals):
            e_acc = e_accs[pl] / e_accs[1]
            e_reward = n * ((e_acc * r) + ((1 - e_acc) * -r))
            # e_reward = n * (e_acc * r)
            cost = c * costs[pl]
            data.append(
                {
                    "texture": texture,
                    "round": round,
                    "patch_size": pl**2,
                    "expected_accuracy": e_acc,
                    "expected_reward": e_reward,
                    "expected_return": e_reward - cost,
                    "c": c,
                }
            )

data = pd.DataFrame(data)

fig, axs = plt.subplots(2, len(c_vals), sharex=True, sharey=True)
for i, texture in enumerate(["rough", "smooth"]):
    for j in range(len(c_vals)):
        df = data[(data["texture"] == texture) & (data["c"] == c_vals[j])]
        df["expected_return"] = (
            df["expected_return"] - df["expected_return"].min()
        ) / (df["expected_return"].max() - df["expected_return"].min())
        sns.lineplot(
            data=df,
            x="patch_size",
            y="expected_return",
            ax=axs[i][j],
        )
        axs[i][j].set_xscale("log", base=2)
        axs[i][j].set(ylim=(0, 1))
        # axs[i][j].axis("off")
fig.tight_layout()
plt.show()

fig, axs = plt.subplots(1, 2)
for i, texture in enumerate(["rough", "smooth"]):
    sns.lineplot(
        data=data[data["texture"] == texture],
        x="patch_size",
        y="expected_accuracy",
        ax=axs[i],
    )
    axs[i].set_xscale("log", base=2)
fig.tight_layout()
plt.show()
