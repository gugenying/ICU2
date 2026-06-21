"""
Streamlit Web App - ML Model Prediction & Visualization

Usage:
  streamlit run 6_app.py
"""

from pathlib import Path
import warnings

import joblib
import matplotlib
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.patches import Patch
from sklearn.exceptions import InconsistentVersionWarning
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix as sk_cm,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.pipeline import Pipeline

matplotlib.use("Agg")
import matplotlib.pyplot as plt


warnings.filterwarnings("ignore", category=UserWarning, module="shap")
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings(
    "ignore",
    message=".*serialized model.*",
    category=UserWarning,
    module="xgboost.*",
)


# ====================================================================
# Configuration
# ====================================================================
BASE_DIR = Path(__file__).parent
MODEL_DIR = BASE_DIR / "output" / "1_models"
SHAP_DIR = BASE_DIR / "output" / "6_shap"
TRAIN_FILE = BASE_DIR / "train1.xlsx"
TEST_FILE = BASE_DIR / "test1.xlsx"
VAL_FILE = BASE_DIR / "val1.xlsx"
LABEL_COL = "Group"
CLASS_IDX = 1
SHAP_SAMPLE_SIZE = 250
APP_TITLE_CN = "基于护理相关暴露因素的ICU患者获得多重耐药菌风险预测模型构建及应用研究"
APP_TITLE_EN = (
    "Development and Application of a Risk Prediction Model for Multidrug-Resistant "
    "Organism Acquisition in ICU Patients Based on Nursing-Related Exposure Factors"
)
APP_FOOTNOTE = "顾艮莹，南京医科大学附属明基医院，13770730245"

SPLIT_LABELS = {
    "train": "训练集 Training",
    "test": "内部验证集 Internal Validation",
    "val": "外部验证集 External Validation",
}

CHART_SPLIT_LABELS = {
    "train": "Training",
    "test": "Internal Validation",
    "val": "External Validation",
}

FEATURE_LABELS = {
    "icu_days": "ICU住院天数（ICU days）",
    "longtermcare_facility_residency": "长期护理机构居住史（Long-term care facility residency）",
    "surgery": "手术史（Surgery）",
    "number_of_surgeries": "手术次数（Number of surgeries）",
    "plt": "血小板计数（Platelet count）",
    "pt": "凝血酶原时间（Prothrombin time）",
    "mechanical_ventilation_days": "机械通气天数（Mechanical ventilation days）",
    "ureter_days": "导尿管留置天数（Ureter catheter days）",
    "antibiotics_days": "抗菌药物使用天数（Antibiotics days）",
    "sedative_drugs_days": "镇静药物使用天数（Sedative drugs days）",
}

FEATURE_CHART_LABELS = {
    "icu_days": "ICU days",
    "longtermcare_facility_residency": "Long-term care facility residency",
    "surgery": "Surgery",
    "number_of_surgeries": "Number of surgeries",
    "plt": "Platelet count",
    "pt": "Prothrombin time",
    "mechanical_ventilation_days": "Mechanical ventilation days",
    "ureter_days": "Ureter catheter days",
    "antibiotics_days": "Antibiotics days",
    "sedative_drugs_days": "Sedative drugs days",
}

MODEL_LABELS = {
    "ABC": "AdaBoost",
    "GBC": "Gradient Boosting",
    "LGB": "LightGBM",
    "LR": "Logistic Regression",
    "RF": "Random Forest",
    "XGB": "XGBoost",
}

TREE_TYPE_NAMES = (
    "RandomForestClassifier",
    "GradientBoostingClassifier",
    "XGBClassifier",
    "LGBMClassifier",
    "DecisionTreeClassifier",
    "ExtraTreesClassifier",
)
LINEAR_TYPE_NAMES = ("LogisticRegression", "LinearSVC", "SGDClassifier")


# ====================================================================
# Page Setup
# ====================================================================
st.set_page_config(
    page_title="ICU多重耐药菌风险预测模型",
    page_icon="🧬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
    :root {
        --ink: #18212f;
        --muted: #657286;
        --line: #dfe7ef;
        --panel: #f7fafc;
        --blue: #2563eb;
        --teal: #0f766e;
        --amber: #b45309;
        --red: #dc2626;
        --green: #15803d;
    }

    .block-container {
        padding-top: 1.4rem;
        padding-bottom: 2.4rem;
        max-width: 1320px;
    }

    [data-testid="stSidebar"] {
        background: #f8fafc;
        border-right: 1px solid var(--line);
    }

    .app-kicker {
        color: var(--teal);
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }

    .app-title {
        color: var(--ink);
        font-size: 1.86rem;
        font-weight: 760;
        line-height: 1.28;
        margin-bottom: 0.3rem;
    }

    .app-title-en {
        color: #3f536f;
        font-size: 0.92rem;
        line-height: 1.45;
        margin-bottom: 0.55rem;
        max-width: 940px;
    }

    .app-subtitle {
        color: var(--muted);
        font-size: 0.98rem;
        margin-bottom: 1.1rem;
        max-width: 860px;
    }

    .section-title {
        color: var(--ink);
        font-size: 1.05rem;
        font-weight: 720;
        margin: 1.05rem 0 0.72rem;
    }

    .result-strip {
        align-items: center;
        background: white;
        border: 1px solid var(--line);
        border-radius: 8px;
        display: grid;
        gap: 0;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        margin-bottom: 1rem;
        overflow: hidden;
    }

    .result-cell {
        border-right: 1px solid var(--line);
        min-height: 118px;
        padding: 1rem 1.1rem;
    }

    .result-cell:last-child {
        border-right: 0;
    }

    .result-label {
        color: var(--muted);
        font-size: 0.76rem;
        font-weight: 700;
        margin-bottom: 0.25rem;
        text-transform: uppercase;
    }

    .result-value {
        color: var(--ink);
        font-size: 2rem;
        font-weight: 780;
        line-height: 1.1;
    }

    .result-note {
        color: var(--muted);
        font-size: 0.78rem;
        margin-top: 0.35rem;
    }

    .status-high { color: var(--red); }
    .status-mid { color: var(--amber); }
    .status-low { color: var(--green); }
    .status-pos { color: var(--red); }
    .status-neg { color: var(--green); }

    .panel {
        background: white;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 1rem;
    }

    .small-muted {
        color: var(--muted);
        font-size: 0.82rem;
    }

    .footer {
        color: #8792a2;
        font-size: 0.78rem;
        line-height: 1.7;
        padding-top: 0.8rem;
        text-align: center;
    }

    div[data-testid="stMetric"] {
        background: white;
        border: 1px solid var(--line);
        border-radius: 8px;
        padding: 0.75rem 0.9rem;
    }

    @media (max-width: 900px) {
        .result-strip {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }

        .result-cell {
            border-bottom: 1px solid var(--line);
        }
    }
</style>
""",
    unsafe_allow_html=True,
)


# ====================================================================
# Helper Functions
# ====================================================================
def display_model_name(model_name: str) -> str:
    return MODEL_LABELS.get(model_name, model_name)


def display_feature_name(feature_name: str) -> str:
    return FEATURE_LABELS.get(feature_name, feature_name.replace("_", " ").title())


def display_chart_feature_name(feature_name: str) -> str:
    return FEATURE_CHART_LABELS.get(feature_name, feature_name.replace("_", " ").title())


def format_feature_value(value) -> str:
    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return str(value)

    if value_float.is_integer():
        return f"{int(value_float)}"
    return f"{value_float:.2f}".rstrip("0").rstrip(".")


def all_integer_values(values) -> bool:
    return all(float(value).is_integer() for value in values)


def aligned_slider_value(min_value, max_value, value, step):
    steps = round((value - min_value) / step)
    aligned = min_value + steps * step
    aligned = min(max(aligned, min_value), max_value)
    decimals = 0 if float(step).is_integer() else 2
    return round(aligned, decimals)


def get_estimator(model):
    if isinstance(model, Pipeline):
        return list(model.named_steps.values())[-1]
    return model


def extract_shap_2d(shap_values, expected_value, class_idx=CLASS_IDX):
    if hasattr(shap_values, "values"):
        shap_values = shap_values.values

    if isinstance(shap_values, list):
        pick_idx = class_idx if len(shap_values) > class_idx else 0
        sv = shap_values[pick_idx]
        ev = (
            expected_value[pick_idx]
            if isinstance(expected_value, (list, tuple, np.ndarray))
            else expected_value
        )
    elif np.ndim(shap_values) == 3:
        pick_idx = class_idx if shap_values.shape[-1] > class_idx else 0
        sv = shap_values[:, :, pick_idx]
        ev = (
            expected_value[pick_idx]
            if isinstance(expected_value, (list, tuple, np.ndarray))
            else expected_value
        )
    else:
        sv = shap_values
        ev = expected_value

    return np.asarray(sv), float(np.asarray(ev).reshape(-1)[0])


def get_available_models():
    if not MODEL_DIR.exists():
        return []

    return sorted(
        file.stem.replace("_best_model", "")
        for file in MODEL_DIR.glob("*_best_model.pkl")
    )


def prediction_frame(model, x_data):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(x_data)[:, CLASS_IDX]
    if hasattr(model, "decision_function"):
        scores = model.decision_function(x_data)
        return 1 / (1 + np.exp(-scores))
    raise ValueError("Selected model does not expose predict_proba or decision_function.")


@st.cache_resource(show_spinner=False)
def load_shap_module():
    try:
        import shap as shap_module
    except Exception as exc:
        raise RuntimeError(f"SHAP dependency is unavailable: {exc}") from exc
    return shap_module


@st.cache_data(show_spinner=False)
def load_data():
    missing = [p.name for p in (TRAIN_FILE, TEST_FILE) if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required data file(s): {', '.join(missing)}")

    train = pd.read_excel(TRAIN_FILE)
    test = pd.read_excel(TEST_FILE)
    datasets = {"train": train, "test": test}
    if VAL_FILE.exists():
        datasets["val"] = pd.read_excel(VAL_FILE)

    for split, df in datasets.items():
        if LABEL_COL not in df.columns:
            raise ValueError(f"{split} data is missing label column '{LABEL_COL}'.")

    return datasets


@st.cache_resource(show_spinner=False)
def load_model(model_name):
    path = MODEL_DIR / f"{model_name}_best_model.pkl"
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path.name}")
    return joblib.load(path)


@st.cache_data(show_spinner=False)
def find_optimal_threshold(y_true, y_proba):
    fpr, tpr, thresholds = roc_curve(y_true, y_proba)
    idx = int(np.argmax(tpr - fpr))
    return float(thresholds[idx])


@st.cache_data(show_spinner=False)
def bootstrap_auc(y_true, y_score, n_bootstraps=300, seed=42):
    y_true = np.asarray(y_true)
    y_score = np.asarray(y_score)

    rng = np.random.RandomState(seed)
    boot = []
    for _ in range(n_bootstraps):
        idx = rng.randint(0, len(y_score), len(y_score))
        if len(np.unique(y_true[idx])) < 2:
            continue
        boot.append(roc_auc_score(y_true[idx], y_score[idx]))

    auc = float(roc_auc_score(y_true, y_score))
    if not boot:
        return auc, np.nan, np.nan

    boot = np.sort(boot)
    return auc, float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


@st.cache_data(show_spinner=False)
def build_feature_profile(train_records, feature_names):
    train_df = pd.DataFrame(train_records, columns=feature_names)
    profile = {}
    for feat in feature_names:
        col = train_df[feat].dropna()
        unique_values = sorted(col.unique())
        profile[feat] = {
            "min": float(col.min()),
            "max": float(col.max()),
            "mean": float(col.mean()),
            "median": float(col.median()),
            "unique": unique_values,
        }
    return profile


@st.cache_data(show_spinner=False)
def get_model_scores(model_name, datasets):
    model = load_model(model_name)
    scores = {}
    for split, df in datasets.items():
        x_data = df.drop(columns=[LABEL_COL])
        y_true = df[LABEL_COL].values
        y_prob = prediction_frame(model, x_data)
        scores[split] = float(roc_auc_score(y_true, y_prob))
    return scores


def get_best_model_name(datasets):
    best_model, best_score = None, -np.inf
    for model_name in get_available_models():
        try:
            scores = get_model_scores(model_name, datasets)
            score = scores.get("val", scores.get("test", np.nan))
            if np.isfinite(score) and score > best_score:
                best_model, best_score = model_name, score
        except Exception:
            continue
    return best_model


@st.cache_resource(show_spinner=False)
def compute_shap_artifacts(model_name):
    shap_module = load_shap_module()
    datasets = load_data()
    x_train = datasets["train"].drop(columns=[LABEL_COL])
    x_explain = x_train.sample(
        n=min(SHAP_SAMPLE_SIZE, len(x_train)),
        random_state=42,
    )
    feature_names = x_train.columns.tolist()
    model = load_model(model_name)
    estimator = get_estimator(model)
    estimator_type = type(estimator).__name__

    if estimator_type in TREE_TYPE_NAMES:
        try:
            explainer = shap_module.TreeExplainer(
                estimator,
                data=x_explain,
                feature_perturbation="interventional",
            )
        except Exception:
            explainer = shap_module.TreeExplainer(estimator)
        raw_values = explainer.shap_values(x_explain)
        values, expected_value = extract_shap_2d(raw_values, explainer.expected_value)
    elif estimator_type in LINEAR_TYPE_NAMES or hasattr(estimator, "coef_"):
        explainer = shap_module.LinearExplainer(estimator, x_explain)
        raw_values = explainer.shap_values(x_explain)
        values, expected_value = extract_shap_2d(raw_values, explainer.expected_value)
    else:
        background = shap_module.kmeans(x_explain, min(20, len(x_explain)))

        def predict_fn(array):
            frame = pd.DataFrame(array, columns=feature_names)
            return prediction_frame(model, frame)

        explainer = shap_module.KernelExplainer(predict_fn, background)
        raw_values = explainer.shap_values(x_explain, nsamples=40, silent=True)
        values, expected_value = extract_shap_2d(raw_values, explainer.expected_value)

    return {
        "explainer": explainer,
        "values": values,
        "expected_value": expected_value,
        "x_explain": x_explain,
        "feature_names": feature_names,
    }


def compute_input_shap(artifacts, input_df):
    explainer = artifacts["explainer"]
    raw_values = explainer.shap_values(input_df)
    values, expected_value = extract_shap_2d(raw_values, explainer.expected_value)
    return values[0], expected_value


def risk_level(probability):
    if probability >= 0.70:
        return "高风险", "status-high"
    if probability >= 0.40:
        return "中风险", "status-mid"
    return "低风险", "status-low"


def metric_row(y_true, y_prob, threshold, split_name):
    y_pred = (y_prob >= threshold).astype(int)
    tn, fp, fn, tp = sk_cm(y_true, y_pred).ravel()
    auc_value = roc_auc_score(y_true, y_prob)
    sensitivity = recall_score(y_true, y_pred, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) else 0
    ppv = precision_score(y_true, y_pred, zero_division=0)
    npv = tn / (tn + fn) if (tn + fn) else 0

    return {
        "数据集 Dataset": split_name,
        "AUC": f"{auc_value:.3f}",
        "灵敏度 Sensitivity": f"{sensitivity:.3f}",
        "特异度 Specificity": f"{specificity:.3f}",
        "PPV": f"{ppv:.3f}",
        "NPV": f"{npv:.3f}",
        "准确率 Accuracy": f"{accuracy_score(y_true, y_pred):.3f}",
        "F1": f"{f1_score(y_true, y_pred, zero_division=0):.3f}",
        "阈值 Threshold": f"{threshold:.4f}",
    }


def draw_probability_gauge(probability, threshold):
    fig, ax = plt.subplots(figsize=(8, 1.45))
    ax.barh([0], [probability], color="#2563eb", height=0.44)
    ax.barh([0], [1 - probability], left=[probability], color="#e8eef5", height=0.44)
    ax.axvline(threshold, color="#dc2626", lw=2, linestyle="--")
    ax.text(
        probability,
        0.29,
        f"{probability:.1%}",
        ha="center",
        va="bottom",
        color="#18212f",
        fontweight="bold",
        fontsize=11,
    )
    ax.text(
        threshold,
        -0.31,
        f"Youden {threshold:.3f}",
        ha="center",
        va="top",
        color="#dc2626",
        fontsize=9,
    )
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.45, 0.48)
    ax.set_xlabel("Predicted probability", fontsize=9)
    ax.set_yticks([])
    for spine in ("top", "right", "left"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", alpha=0.18)
    fig.tight_layout()
    return fig


def draw_roc_curve(y_true, y_prob, title):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc_value, lo, hi = bootstrap_auc(y_true, y_prob)
    label = f"AUC = {auc_value:.3f}"
    if np.isfinite(lo) and np.isfinite(hi):
        label += f"\n95% CI: {lo:.3f}-{hi:.3f}"

    fig, ax = plt.subplots(figsize=(4.35, 4.25))
    ax.plot(fpr, tpr, lw=2.4, color="#2563eb", label=label)
    ax.fill_between(fpr, tpr, alpha=0.08, color="#2563eb")
    ax.plot([0, 1], [0, 1], color="#94a3b8", linestyle="--", lw=1)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title, fontsize=11, fontweight="bold", color="#18212f")
    ax.legend(loc="lower right", fontsize=8, frameon=True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(alpha=0.18)
    fig.tight_layout()
    return fig


def draw_shap_bar(shap_values, input_df, feature_names, expected_value):
    order = np.argsort(np.abs(shap_values))[::-1]
    sorted_names = [feature_names[i] for i in order]
    sorted_values = shap_values[order]
    feature_values = [input_df.iloc[0][name] for name in sorted_names]

    fig_height = max(4.6, len(sorted_names) * 0.48 + 1.2)
    fig, ax = plt.subplots(figsize=(9.4, fig_height))
    colors = ["#dc2626" if value > 0 else "#2563eb" for value in sorted_values]
    y_pos = np.arange(len(sorted_names))
    bars = ax.barh(y_pos, sorted_values, height=0.55, color=colors, edgecolor="none")
    max_abs_value = max(float(np.max(np.abs(sorted_values))), 0.01)
    ax.set_xlim(-max_abs_value * 1.35, max_abs_value * 1.35)

    y_labels = [
        f"{display_chart_feature_name(name)} = {format_feature_value(value)}"
        for name, value in zip(sorted_names, feature_values)
    ]
    ax.set_yticks(y_pos)
    ax.set_yticklabels(y_labels, fontsize=9)

    for bar, value in zip(bars, sorted_values):
        offset = 0.003 if value >= 0 else -0.003
        ax.text(
            value + offset,
            bar.get_y() + bar.get_height() / 2,
            f"{value:+.3f}",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.4,
            color="#334155",
        )

    ax.axvline(0, color="#475569", linewidth=1.1)
    ax.set_xlabel("SHAP value (impact on model output)", fontsize=9)
    ax.set_title(
        f"Base value {expected_value:.3f} | model output {expected_value + shap_values.sum():.3f}",
        fontsize=10,
        color="#334155",
        pad=8,
    )
    ax.legend(
        handles=[
            Patch(facecolor="#dc2626", label="Increases risk"),
            Patch(facecolor="#2563eb", label="Decreases risk"),
        ],
        loc="lower right",
        fontsize=8.5,
        framealpha=0.88,
    )
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(axis="x", linestyle="--", alpha=0.25)
    ax.invert_yaxis()
    fig.tight_layout()
    return fig


# ====================================================================
# Main App
# ====================================================================
try:
    datasets = load_data()
except Exception as exc:
    st.error(f"Data loading failed: {exc}")
    st.stop()

available_models = get_available_models()
if not available_models:
    st.error("No trained models found in output/1_models/. Please run 1_multimodel.py first.")
    st.stop()

train_df = datasets["train"]
x_train_full = train_df.drop(columns=[LABEL_COL])
y_train_full = train_df[LABEL_COL].values
feature_names = x_train_full.columns.tolist()
feature_profile = build_feature_profile(
    x_train_full.to_dict("records"),
    feature_names,
)
available_splits = [split for split in SPLIT_LABELS if split in datasets]
best_model_name = get_best_model_name(datasets)


# ====================================================================
# Sidebar
# ====================================================================
with st.sidebar:
    st.markdown("### 模型选择")
    model_choice = st.selectbox(
        "已训练模型",
        options=available_models,
        index=available_models.index(best_model_name)
        if best_model_name in available_models
        else 0,
        format_func=display_model_name,
        help="默认优先选择验证集AUC表现较好的模型。",
    )

    if model_choice == best_model_name:
        st.caption("当前为验证集表现较优模型")

    st.markdown("### 患者护理相关暴露因素")
    st.caption("连续变量默认采用训练集中的中位数，可按实际患者情况调整。")

    input_values = {}
    for feature in feature_names:
        stats = feature_profile[feature]
        unique_values = stats["unique"]
        label = display_feature_name(feature)

        if set(unique_values).issubset({0, 1}):
            default_index = 1 if stats["median"] >= 0.5 else 0
            input_values[feature] = st.selectbox(
                label,
                options=[0, 1],
                index=default_index,
                format_func=lambda value: "是" if value == 1 else "否",
                key=f"feat_{feature}",
            )
        elif len(unique_values) <= 8 and all_integer_values(unique_values):
            integer_values = [int(v) for v in unique_values]
            median_value = int(round(stats["median"]))
            input_values[feature] = st.selectbox(
                label,
                options=integer_values,
                index=integer_values.index(median_value)
                if median_value in integer_values
                else 0,
                key=f"feat_{feature}",
            )
        elif all_integer_values(unique_values):
            input_values[feature] = st.slider(
                label,
                min_value=int(round(stats["min"])),
                max_value=int(round(stats["max"])),
                value=int(round(stats["median"])),
                step=1,
                key=f"feat_{feature}",
            )
        else:
            span = max(stats["max"] - stats["min"], 1.0)
            step = max(0.01, round(span / 200, 2))
            min_value = round(stats["min"], 2)
            max_value = round(stats["max"], 2)
            default_value = aligned_slider_value(
                min_value,
                max_value,
                round(stats["median"], 2),
                step,
            )
            input_values[feature] = st.slider(
                label,
                min_value=min_value,
                max_value=max_value,
                value=default_value,
                step=step,
                key=f"feat_{feature}",
            )


# ====================================================================
# Header and model predictions
# ====================================================================
st.markdown('<div class="app-kicker">ICU多重耐药菌获得风险预测系统</div>', unsafe_allow_html=True)
st.markdown(f'<div class="app-title">{APP_TITLE_CN}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="app-title-en">{APP_TITLE_EN}</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">'
    "本网页用于基于护理相关暴露因素进行ICU患者多重耐药菌获得风险的个体化预测，"
    "并展示模型验证ROC曲线、SHAP解释及模型性能指标。"
    "</div>",
    unsafe_allow_html=True,
)

try:
    model = load_model(model_choice)
    input_df = pd.DataFrame([input_values], columns=feature_names)
    probability = float(prediction_frame(model, input_df)[0])
    train_probability = prediction_frame(model, x_train_full)
    threshold = find_optimal_threshold(y_train_full, train_probability)
except Exception as exc:
    st.error(f"Prediction failed for {display_model_name(model_choice)}: {exc}")
    st.stop()

prediction = int(probability >= threshold)
risk_text, risk_class = risk_level(probability)
decision_text = "阳性" if prediction else "阴性"
decision_class = "status-pos" if prediction else "status-neg"
test_auc = get_model_scores(model_choice, datasets).get("test", np.nan)

st.markdown(
    f"""
<div class="result-strip">
    <div class="result-cell">
        <div class="result-label">预测概率</div>
        <div class="result-value">{probability:.1%}</div>
        <div class="result-note">多重耐药菌获得风险概率</div>
    </div>
    <div class="result-cell">
        <div class="result-label">风险等级</div>
        <div class="result-value {risk_class}">{risk_text}</div>
        <div class="result-note">依据预测概率分层</div>
    </div>
    <div class="result-cell">
        <div class="result-label">模型判定</div>
        <div class="result-value {decision_class}">{decision_text}</div>
        <div class="result-note">Youden阈值 {threshold:.3f}</div>
    </div>
    <div class="result-cell">
        <div class="result-label">内部验证AUC</div>
        <div class="result-value">{test_auc:.3f}</div>
        <div class="result-note">{display_model_name(model_choice)}</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)


# ====================================================================
# Tabs
# ====================================================================
tab_prediction, tab_roc, tab_shap, tab_metrics = st.tabs(
    ["个体预测", "ROC曲线", "SHAP解释", "模型指标"]
)


# ====================================================================
# Tab 1: Prediction
# ====================================================================
with tab_prediction:
    col_gauge, col_inputs = st.columns([1.8, 1])

    with col_gauge:
        st.markdown('<div class="section-title">预测概率展示</div>', unsafe_allow_html=True)
        fig = draw_probability_gauge(probability, threshold)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    with col_inputs:
        st.markdown('<div class="section-title">输入变量汇总</div>', unsafe_allow_html=True)
        summary_df = pd.DataFrame(
            {
                "变量 Feature": [display_feature_name(name) for name in feature_names],
                "取值 Value": [format_feature_value(input_values[name]) for name in feature_names],
            }
        )
        st.dataframe(summary_df, use_container_width=True, height=295, hide_index=True)

    st.markdown('<div class="section-title">个体预测SHAP贡献</div>', unsafe_allow_html=True)
    try:
        with st.spinner("正在计算个体SHAP解释..."):
            artifacts = compute_shap_artifacts(model_choice)
            shap_values, expected_value = compute_input_shap(artifacts, input_df)
            fig = draw_shap_bar(shap_values, input_df, feature_names, expected_value)
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)
    except Exception as exc:
        st.info(f"当前模型暂无法展示SHAP解释：{exc}")


# ====================================================================
# Tab 2: ROC Curves
# ====================================================================
with tab_roc:
    st.markdown('<div class="section-title">模型验证ROC曲线</div>', unsafe_allow_html=True)
    cols = st.columns(len(available_splits))
    for col, split in zip(cols, available_splits):
        with col:
            df_split = datasets[split]
            y_true = df_split[LABEL_COL].values
            x_feat = df_split.drop(columns=[LABEL_COL])
            y_prob = prediction_frame(model, x_feat)
            fig = draw_roc_curve(y_true, y_prob, CHART_SPLIT_LABELS[split])
            st.pyplot(fig, use_container_width=True)
            plt.close(fig)


# ====================================================================
# Tab 3: SHAP Analysis
# ====================================================================
with tab_shap:
    st.markdown('<div class="section-title">全局特征重要性</div>', unsafe_allow_html=True)
    png_dot = SHAP_DIR / model_choice / f"{model_choice}_shap_summary_dot.png"
    png_bar = SHAP_DIR / model_choice / f"{model_choice}_shap_summary_bar.png"

    col_left, col_right = st.columns(2)
    with col_left:
        st.markdown("**SHAP蜂群图（Bee Swarm Plot）**")
        if png_dot.exists():
            st.image(str(png_dot), use_container_width=True)
        else:
            try:
                shap_module = load_shap_module()
                artifacts = compute_shap_artifacts(model_choice)
                fig, _ = plt.subplots(figsize=(7.2, max(4, len(feature_names) * 0.42)))
                shap_module.summary_plot(
                    artifacts["values"],
                    artifacts["x_explain"],
                    feature_names=[display_chart_feature_name(f) for f in feature_names],
                    plot_type="dot",
                    show=False,
                )
                plt.tight_layout()
                st.pyplot(plt.gcf(), use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.info(f"蜂群图暂不可用：{exc}")

    with col_right:
        st.markdown("**平均绝对SHAP值（Mean |SHAP|）**")
        if png_bar.exists():
            st.image(str(png_bar), use_container_width=True)
        else:
            try:
                shap_module = load_shap_module()
                artifacts = compute_shap_artifacts(model_choice)
                fig, _ = plt.subplots(figsize=(7.2, max(4, len(feature_names) * 0.42)))
                shap_module.summary_plot(
                    artifacts["values"],
                    artifacts["x_explain"],
                    feature_names=[display_chart_feature_name(f) for f in feature_names],
                    plot_type="bar",
                    show=False,
                )
                plt.tight_layout()
                st.pyplot(plt.gcf(), use_container_width=True)
                plt.close(fig)
            except Exception as exc:
                st.info(f"SHAP条形图暂不可用：{exc}")

    st.markdown('<div class="section-title">SHAP特征重要性表</div>', unsafe_allow_html=True)
    try:
        artifacts = compute_shap_artifacts(model_choice)
        mean_shap = np.abs(artifacts["values"]).mean(axis=0)
        shap_df = (
            pd.DataFrame(
                {
                    "变量 Feature": [display_feature_name(name) for name in feature_names],
                    "Mean |SHAP|": mean_shap,
                }
            )
            .sort_values("Mean |SHAP|", ascending=False)
            .reset_index(drop=True)
        )
        shap_df.insert(0, "排序 Rank", np.arange(1, len(shap_df) + 1))
        st.dataframe(
            shap_df.style.background_gradient(subset=["Mean |SHAP|"], cmap="Blues"),
            use_container_width=True,
            hide_index=True,
        )
    except Exception as exc:
        st.info(f"SHAP表格暂不可用：{exc}")


# ====================================================================
# Tab 4: Metrics Table
# ====================================================================
with tab_metrics:
    st.markdown('<div class="section-title">当前模型性能</div>', unsafe_allow_html=True)
    selected_rows = []
    for split in available_splits:
        df_split = datasets[split]
        y_true = df_split[LABEL_COL].values
        y_prob = prediction_frame(model, df_split.drop(columns=[LABEL_COL]))
        selected_rows.append(metric_row(y_true, y_prob, threshold, SPLIT_LABELS[split]))

    selected_metrics = pd.DataFrame(selected_rows)
    st.dataframe(selected_metrics.set_index("数据集 Dataset"), use_container_width=True)

    st.markdown('<div class="section-title">全部模型AUC比较</div>', unsafe_allow_html=True)
    auc_rows = []
    for model_name in available_models:
        try:
            row = {"模型 Model": display_model_name(model_name)}
            candidate = load_model(model_name)
            for split in available_splits:
                df_split = datasets[split]
                y_true = df_split[LABEL_COL].values
                y_prob = prediction_frame(candidate, df_split.drop(columns=[LABEL_COL]))
                auc_value, lo, hi = bootstrap_auc(y_true, y_prob)
                if np.isfinite(lo) and np.isfinite(hi):
                    row[SPLIT_LABELS[split]] = f"{auc_value:.3f} ({lo:.3f}-{hi:.3f})"
                else:
                    row[SPLIT_LABELS[split]] = f"{auc_value:.3f}"
            auc_rows.append(row)
        except Exception:
            continue

    if auc_rows:
        auc_df = pd.DataFrame(auc_rows).set_index("模型 Model")
        st.dataframe(auc_df, use_container_width=True)
    else:
        st.info("暂无法计算模型指标。")


st.markdown("---")
st.markdown(
    f'<div class="footer">{APP_FOOTNOTE}<br>ICU多重耐药菌风险预测模型 | scikit-learn + SHAP + Streamlit</div>',
    unsafe_allow_html=True,
)
