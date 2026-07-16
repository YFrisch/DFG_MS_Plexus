from pathlib import Path

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold, RepeatedStratifiedKFold, cross_validate, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report, confusion_matrix, make_scorer, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight, compute_class_weight

from imblearn.pipeline import Pipeline as ImblearnPipeline
from imblearn.over_sampling import SMOTE, ADASYN
from imblearn.ensemble import BalancedBaggingClassifier, BalancedRandomForestClassifier
from imblearn.combine import SMOTEENN

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool

from .labels import get_labels_full, get_labels_hc_cis_ms, get_labels_hc_ms


def aggregate_reports(reports):
    stacked = pd.concat(
        reports,
        keys=range(len(reports)),
        names=["seed", "metric"],
    )

    mean = stacked.groupby(level="metric").mean()
    std = stacked.groupby(level="metric").std()

    formatted = mean.copy()

    for col in ["precision", "recall", "f1-score"]:
        formatted[col] = [
            f"{m:.3f} ± {s:.3f}"
            for m, s in zip(mean[col], std[col])
        ]

    # Support is usually better reported as mean count, not ± std,
    # because each seed has a different split.
    formatted["support"] = [
        f"{m:.1f} ± {s:.1f}"
        for m, s in zip(mean["support"], std["support"])
    ]

    return formatted


def report_to_df(report_dict):
    df_report = pd.DataFrame(report_dict).T
    return df_report[["precision", "recall", "f1-score", "support"]]


def normalize_cm(cm):
    row_sums = cm.sum(axis=1, keepdims=True)
    return np.divide(
        cm,
        row_sums,
        out=np.zeros_like(cm, dtype=float),
        where=row_sums != 0,
    )


def plot_mean_std_cm(
        cm_mean,
        cm_std,
        title,
        class_names,
        save_path: str | None = None
):
    annot = np.empty_like(cm_mean, dtype=object)

    for i in range(cm_mean.shape[0]):
        for j in range(cm_mean.shape[1]):
            annot[i, j] = f"{cm_mean[i, j]:.2f}\n± {cm_std[i, j]:.2f}"

    plt.figure(figsize=(7, 6))
    sns.heatmap(
        cm_mean,
        annot=annot,
        fmt="",
        cmap="Reds",
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0,
        vmax=1,
    )
    plt.title(title)
    plt.xlabel("Predicted label")
    plt.ylabel("True label")
    plt.tight_layout()

    if save_path is not None:
        if ".svg" in save_path:
            plt.rcParams["svg.fonttype"] = "none"
            plt.savefig(save_path, bbox_inches="tight", format="svg")
        else:
            plt.savefig(save_path, bbox_inches="tight", format="png")

    plt.show()
