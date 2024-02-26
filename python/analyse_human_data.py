from collections import defaultdict
from itertools import product
import json
import sqlite3

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from src.utils import save_figure, to_range
from src.modelling import expected_acc_approx

NUM_ROUNDS = 5
NUM_CHOICES = 20
CHOICE_REWARD = 5
MAP_COSTS = {
    1: 100,
    2: 40,
    4: 10,
    8: 1,
}
COST_MULTIPLIERS = [0.1, 1, 5]
TEXTURES = ["rough", "smooth"]
BETA = 0.3

sns.set_theme()
sns.set_context("paper")


def load_sessions(cursor):
    res = cur.execute("SELECT id,experiment_id,user_id,texture,cost FROM sessions")
    sessions = res.fetchall()
    return {
        s[0]: {
            "id": s[0],
            "experiment_id": s[1],
            "user_id": s[2],
            "texture": s[3],
            "cost": s[4],
        }
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


def load_heatmaps():
    hmaps = defaultdict(lambda: defaultdict(dict))
    for txtr, rnd, pl in product(TEXTURES, range(1, 11), MAP_COSTS.keys()):
        with open(f"../heatmaps/{txtr}/{rnd}/{pl}.txt", "r") as f:
            hmap = np.array(json.load(f))
            hmaps[txtr][rnd][pl] = np.repeat(np.repeat(hmap, pl, axis=0), pl, axis=1)
    return hmaps


def process_human_data(session_dict, choice_dict, heatmaps):
    # compute participant accuracies
    acc_data = []
    for sid, choices in choice_dict.items():
        # if len(choices) != NUM_ROUNDS * NUM_CHOICES:
        #     continue

        accs, pss = {}, {}
        sess = session_dict[sid]
        hmaps = heatmaps[sess["texture"]]

        for c in choices:
            if c["selected"] not in (0, 1):
                continue

            vals = np.array([hmaps[c["round"]][1][c[f"tile_{i}"]] for i in (1, 2)])
            delta = vals[c["selected"]] - vals[1 - c["selected"]]
            if delta > 0:
                correct = 1
            elif delta < 0:
                correct = 0
            else:
                correct = 0.5

            accs[c["round"]] = accs.get(c["round"], 0) + correct
            if c["round"] not in pss:
                pss[c["round"]] = c["patch_size"]

        for round in accs:
            acc_data.append(
                {
                    "session_id": sid,
                    "texture": sess["texture"],
                    "c": sess["cost"],
                    "round": round,
                    "patch_size": pss[round] ** 2,
                    "accuracy": accs[round] / NUM_CHOICES,
                }
            )

    # compute patch size choice distribution from human data
    acc_data = pd.DataFrame(acc_data)
    ps_counts = acc_data.groupby(["texture", "c", "patch_size"]).size()
    ps_counts = ps_counts.reset_index(name="count")

    # fill in missing counts with 0
    for txtr, cm, ps in product(TEXTURES, COST_MULTIPLIERS, MAP_COSTS.keys()):
        if (
            ps**2
            not in ps_counts[(ps_counts["texture"] == txtr) & (ps_counts["c"] == cm)][
                "patch_size"
            ].values
        ):
            ps_counts.loc[len(ps_counts)] = {
                "texture": txtr,
                "c": cm,
                "patch_size": ps**2,
                "count": 0,
            }

    # return accuracy data and patch size counts
    return acc_data, ps_counts


# load sessions, choices and heatmaps
conn = sqlite3.connect("../remote.db")
cur = conn.cursor()
sessions = load_sessions(cur)
choices = load_choices(cur)
hmaps = load_heatmaps()
conn.close()

# compute accuracy values and patch size counts from human data
human_accs, patch_size_counts = process_human_data(sessions, choices, hmaps)

# get the number of datapoints for each texture and c
print(human_accs.groupby(["texture", "c"]).size())

# compute expected accuracies, returns and patch size choice probabilities from model
model_data, pls = [], list(MAP_COSTS.keys())
# 1,2,6,8
for txtr, rnd, c in product(TEXTURES, range(1, NUM_ROUNDS + 1), COST_MULTIPLIERS):
    accs, rewards, costs, rets = (
        np.zeros(len(pls)),
        np.zeros(len(pls)),
        np.zeros(len(pls)),
        np.zeros(len(pls)),
    )

    for i, pl in enumerate(pls):
        accs[i] = expected_acc_approx(hmaps[txtr][rnd][1], hmaps[txtr][rnd][pl], 0.0001)
        rewards[i] = (accs[i] * CHOICE_REWARD) + ((1 - accs[i]) * -CHOICE_REWARD)
        costs[i] = c * MAP_COSTS[pl]
        rets[i] = NUM_CHOICES * (rewards[i]) - (c * MAP_COSTS[pl])

    acc_choice_probs = np.exp(to_range(accs, 0, 1) / BETA)
    reward_choice_probs = np.exp(to_range(rewards, 0, 1) / BETA)
    cost_choice_probs = np.exp(to_range(-costs, 0, 1) / BETA)
    ret_choice_probs = np.exp(to_range(rets, 0, 1) / BETA)

    model_data.extend(
        [
            {
                "texture": txtr,
                "round": rnd,
                "c": c,
                "patch_size": pl**2,
                "e_accuracy": accs[i],
                "choice_prob_acc": acc_choice_probs[i] / np.sum(acc_choice_probs),
                "choice_prob_reward": reward_choice_probs[i]
                / np.sum(reward_choice_probs),
                "choice_prob_cost": cost_choice_probs[i] / np.sum(cost_choice_probs),
                "choice_prob_ret": ret_choice_probs[i] / np.sum(ret_choice_probs),
            }
            for i, pl in enumerate(pls)
        ]
    )
model_data = pd.DataFrame(model_data)

# # plot accuracy vs patch size
# fig, axs = plt.subplots(2, 1, figsize=(4.5, 5), sharex=True)
# for i, texture in enumerate(TEXTURES):
#     sns.lineplot(
#         data=model_data[model_data["texture"] == texture],
#         x="patch_size",
#         y="e_accuracy",
#         color="#271AB7",
#         ax=axs[i],
#     )
#     sns.lineplot(
#         data=human_accs[human_accs["texture"] == texture],
#         x="patch_size",
#         y="accuracy",
#         color="#009E78",
#         ax=axs[i],
#     )
#     axs[i].set_xscale("log", base=2)
#     axs[i].set_xlabel("Patch size", fontsize=11)
#     axs[i].set_ylabel("Accuracy", fontsize=11)
#     axs[i].set(
#         xticks=[x**2 for x in MAP_COSTS.keys()],
#         xticklabels=[str(int(x**2)) for x in MAP_COSTS.keys()],
#     )
# save_figure(fig, "experiment_1_accuracy")


fig, axs = plt.subplots(
    len(TEXTURES), len(COST_MULTIPLIERS), figsize=(12, 5), sharex=True, sharey=True
)
for i, j in product(range(len(TEXTURES)), range(len(COST_MULTIPLIERS))):
    human_df = patch_size_counts[
        (patch_size_counts["texture"] == TEXTURES[i])
        & (patch_size_counts["c"] == COST_MULTIPLIERS[j])
    ]
    human_df["count"] = human_df["count"] / human_df["count"].sum()
    model_df = model_data[
        (model_data["texture"] == TEXTURES[i])
        & (model_data["c"] == COST_MULTIPLIERS[j])
    ]

    for policy, color, alpha in zip(
        ["acc", "cost", "ret"], ["#D64F5F", "#C68847", "#271AB7"], [0.5, 0.5, 1.0]
    ):
        sns.lineplot(
            data=model_df,
            x="patch_size",
            y=f"choice_prob_{policy}",
            color=color,
            alpha=alpha,
            ax=axs[i][j],
        )
    sns.scatterplot(
        data=human_df, x="patch_size", y="count", color="#009E78", s=65, ax=axs[i][j]
    )
    axs[i][j].set_xscale("log", base=2)
    axs[i][j].set_xlabel("Patch size", fontsize=11)
    axs[i][j].set_ylabel("Choice probability", fontsize=11)
    axs[i][j].set(
        xticks=[x**2 for x in MAP_COSTS.keys()],
        xticklabels=[str(int(x**2)) for x in MAP_COSTS.keys()],
        ylim=(-0.05, 1.05),
    )
save_figure(fig, "experiment_1_choice_probabilities")
