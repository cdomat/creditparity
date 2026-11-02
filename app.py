"""CreditParity – Profitable, Fair Lending Decisions.

Streamlit dashboard for credit analysts. Four sections:
  1. Single application  – default probability, rating, zone and SHAP drivers
  2. Batch upload        – score a CSV of applications and download the results
  3. Bank view           – policy sliders with live workload, cost and fairness metrics
  4. Model info          – model choice, price of fairness and limitations

All models and data are loaded from model/ (created in notebook Step 8).
"""
from pathlib import Path
import json
import re

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
DATA_PATH = ROOT / "data" / "german_credit_data_biased_training.csv"

st.set_page_config(page_title="CreditParity", page_icon="🏦", layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------------------
NAVY = "#0d366b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
MODEL_COLORS = {"d": "#2a78d6", "a": "#eb6834"}          # categorical slots 1 and 2
RAISES_RISK, LOWERS_RISK = "#e34948", "#2a78d6"           # diverging pair (red / blue)
ZONES = {                                                 # status palette + icon + label
    "Green": {"label": "Auto-approve", "color": "#0ca30c", "icon": "✅",
              "text": "Approved automatically."},
    "Yellow": {"label": "Manual review", "color": "#fab219", "icon": "🔎",
               "text": "A credit analyst decides, supported by the explanation below."},
    "Red": {"label": "Decline recommended", "color": "#d03b3b", "icon": "⛔",
            "text": "The system declines with an explanation; the customer can request a human review."},
}
ZONE_ORDER = ["Green", "Yellow", "Red"]

# Fixed master scale: the rating describes the applicant's risk and never moves with the policy.
RATING_SCALE = [("A", 0.00, 0.05), ("B", 0.05, 0.15), ("C", 0.15, 0.35),
                ("D", 0.35, 0.70), ("E", 0.70, 1.01)]

MODEL_NAMES = {"d": "(d) Fair model – final", "a": "(a) All features – comparison"}
NOT_SCORED = ["Sex", "ForeignWorker"]                      # not inputs of model (d)
AGE_LABEL = "Age (fairness correction)"

st.markdown(f"""
<style>
  .block-container {{padding-top: 3.2rem; max-width: 1250px;}}
  h1, h2, h3 {{color: {NAVY};}}
  .cp-header {{display:flex; align-items:baseline; gap:.8rem; margin-bottom:.2rem;}}
  .cp-title {{font-size:2rem; font-weight:700; color:{NAVY}; letter-spacing:-.01em;}}
  .cp-tagline {{font-size:1.05rem; color:{TEXT_SECONDARY};}}
  .cp-badge {{display:inline-flex; align-items:center; gap:.4rem; padding:.25rem .7rem;
             border-radius:999px; font-weight:600; font-size:.95rem; border:1px solid #d9dde3;
             background:#ffffff; color:#0b0b0b;}}
  .cp-dot {{width:.7rem; height:.7rem; border-radius:50%; display:inline-block;}}
  .cp-pd {{font-size:3rem; font-weight:700; color:{NAVY}; line-height:1.1;}}
  .cp-muted {{color:{TEXT_SECONDARY}; font-size:.9rem;}}
  div[data-testid="stMetricValue"] {{color:{NAVY};}}
</style>
""", unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------------------
@st.cache_resource
def load_bundle():
    return joblib.load(MODEL_DIR / "creditparity_bundle.joblib")


@st.cache_data
def load_evaluation():
    return pd.read_csv(MODEL_DIR / "evaluation_data.csv", index_col="application_id")


@st.cache_data
def load_dataset():
    return pd.read_csv(DATA_PATH)


bundle = load_bundle()
POLICY = bundle["policy"]
META = bundle["metadata"]
MODELS = {"d": bundle["final_model"], "a": bundle["comparison_model"]}
EXPLAIN = bundle["explain_data"]
INPUT_FEATURES = META["input_features"]
CATEGORICAL = META["categorical_features"]
evaluation = load_evaluation()
dataset = load_dataset()


# --------------------------------------------------------------------------------------
# Helpers: labels, rating, zones, policy
# --------------------------------------------------------------------------------------
def feature_label(feature):
    if feature == "Age":
        return AGE_LABEL
    return re.sub(r"(?<!^)(?=[A-Z])", " ", feature).capitalize()


def value_label(value):
    """Readable label for raw category codes such as 'less_0' or '0_to_200'."""
    if not isinstance(value, str):
        return str(value)
    v = value.replace("_to_", " – ")
    v = re.sub(r"^less_", "< ", v)
    v = re.sub(r"^greater_", "> ", v)
    v = v.replace("_", " ")
    return v[:1].upper() + v[1:]


def rating_of(p):
    return next(r for r, lo, hi in RATING_SCALE if lo <= p < hi)


def zone_of(p, lower, upper):
    p = np.asarray(p)
    return np.where(p < lower, "Green", np.where(p >= upper, "Red", "Yellow"))


THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


@st.cache_data
def upper_for_precision(model_key, lower, target):
    """Precision rule (Step 5): lowest threshold >= lower at which at least `target`
    of the red-zone applications are defaults, on the out-of-fold training predictions."""
    oof = evaluation[evaluation["split"] == "train_oof"]
    p = oof["pd_final" if model_key == "d" else "pd_comparison"].to_numpy()
    y = oof["y"].to_numpy()
    for t in THRESHOLDS[THRESHOLDS >= lower]:
        red = p >= t
        if red.any() and y[red].mean() >= target:
            return float(t)
    return 1.0


def init_state():
    # Keep form values when the user switches sections (Streamlit drops the state of
    # widgets that are not rendered in a run).
    for k in list(st.session_state.keys()):
        if k.startswith("f_") or k == "preset":
            st.session_state[k] = st.session_state[k]
    for key in MODELS:
        st.session_state.setdefault(f"lower_{key}", POLICY["models"][key]["lower_boundary"])
    st.session_state.setdefault("target", POLICY["target_precision"])
    st.session_state.setdefault("compare", False)


def persist(widget_key, state_key):
    st.session_state[state_key] = st.session_state[widget_key]


def reset_policy():
    for key in MODELS:
        st.session_state[f"lower_{key}"] = POLICY["models"][key]["lower_boundary"]
        st.session_state.pop(f"w_lower_{key}", None)
    st.session_state["target"] = POLICY["target_precision"]
    st.session_state.pop("w_target", None)


def current_policy(model_key):
    lower = (st.session_state[f"lower_{model_key}"] if model_key == "d"
             else POLICY["models"]["a"]["lower_boundary"])
    target = st.session_state["target"]
    return lower, upper_for_precision(model_key, lower, target), target


# --------------------------------------------------------------------------------------
# Scoring and explanations
# --------------------------------------------------------------------------------------
def predict(data, model_key="d"):
    return MODELS[model_key].predict_proba(data[INPUT_FEATURES])[:, 1]


def explain(data, model_key="d"):
    """Exact linear SHAP values (log-odds) per input feature: coefficient x deviation
    from the average training applicant, summed over the one-hot columns of a feature."""
    model, info = MODELS[model_key], EXPLAIN[model_key]
    Z = model.named_steps["prep"].transform(data[INPUT_FEATURES])
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    coef = model.named_steps["model"].coef_[0]
    phi = pd.DataFrame(coef * (Z - info["background_mean"]), columns=info["columns"], index=data.index)
    base = float(model.named_steps["model"].intercept_[0] + coef @ info["background_mean"])
    return phi.T.groupby(info["feature_of_column"], sort=False).sum().T, base


def score_table(data):
    p = predict(data, "d")
    lower, upper, _ = current_policy("d")
    contrib, _ = explain(data, "d")
    main_driver = contrib.idxmax(axis=1).map(feature_label)
    out = data.copy()
    out.insert(0, "Default probability (%)", (100 * p).round(1))
    out.insert(1, "Rating", [rating_of(v) for v in p])
    out.insert(2, "Zone", [f"{ZONES[z]['icon']} {z} – {ZONES[z]['label']}" for z in zone_of(p, lower, upper)])
    out.insert(3, "Main risk driver", main_driver.where(contrib.max(axis=1) > 0, "–"))
    return out


# --------------------------------------------------------------------------------------
# UI building blocks
# --------------------------------------------------------------------------------------
def header():
    st.markdown('<div class="cp-header"><span class="cp-title">🏦 CreditParity</span>'
                '<span class="cp-tagline">Profitable, Fair Lending Decisions</span></div>',
                unsafe_allow_html=True)


def zone_badge(zone, rating=None):
    z = ZONES[zone]
    rating_part = f"Rating {rating} · " if rating else ""
    return (f'<span class="cp-badge"><span class="cp-dot" style="background:{z["color"]}"></span>'
            f'{z["icon"]} {rating_part}{zone} zone – {z["label"]}</span>')


def drivers_chart(contrib, data_row, top=8):
    order = contrib.abs().sort_values(ascending=False).index
    shown = contrib[order[:top]]
    rows = [{"Driver": f"{feature_label(f)} = {value_label(data_row[f])}",
             "Contribution": float(v),
             "Effect": "Raises risk" if v > 0 else "Lowers risk"} for f, v in shown.items()]
    rest = float(contrib[order[top:]].sum())
    rows.append({"Driver": f"{len(order) - top} other features", "Contribution": rest,
                 "Effect": "Raises risk" if rest > 0 else "Lowers risk"})
    df = pd.DataFrame(rows)
    df["Contribution (log-odds)"] = df["Contribution"].round(2)
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=18).encode(
        x=alt.X("Contribution:Q", title="Contribution to risk (log-odds)",
                axis=alt.Axis(gridColor="#ececea", labelColor=TEXT_SECONDARY, titleColor=TEXT_SECONDARY)),
        y=alt.Y("Driver:N", sort=None, title=None, axis=alt.Axis(labelLimit=320, labelColor="#0b0b0b")),
        color=alt.Color("Effect:N", scale=alt.Scale(domain=["Raises risk", "Lowers risk"],
                                                    range=[RAISES_RISK, LOWERS_RISK]),
                        legend=alt.Legend(orient="top", title=None)),
        tooltip=["Driver", "Effect", "Contribution (log-odds)"],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8c8c8c").encode(x="x:Q")
    return (bars + zero).properties(height=34 * len(df)).configure_view(stroke=None)


def credit_memo(row, p, rating, zone, contrib, lower, upper):
    up = contrib[contrib > 0].sort_values(ascending=False).head(3)
    down = contrib[contrib < 0].sort_values().head(3)
    line = lambda f, v: f"  • {feature_label(f)}: {value_label(row[f])} ({v:+.2f})"
    return "\n".join([
        "CREDIT MEMO – CreditParity (analyst use only)",
        "",
        f"Loan: {row['LoanAmount']:,} for {row['LoanDuration']} months ({value_label(row['LoanPurpose'])})",
        f"Default probability: {p:.1%} · Rating {rating} · {zone} zone",
        f"Policy: green below {lower:.0%}, red from {upper:.0%}",
        "",
        "Factors raising the risk:",
        *([line(f, v) for f, v in up.items()] or ["  • none"]),
        "",
        "Factors lowering the risk:",
        *([line(f, v) for f, v in down.items()] or ["  • none"]),
        "",
        "Analyst checklist:",
        "  • Verify income, employment and accounts",
        "  • Are the main risk factors temporary?",
        "  • Age is a fairness correction only,",
        "    never a reason on its own",
        "  • Sex and foreign worker status: not used",
        "",
        "Decision: [ ] approve  [ ] with conditions",
        "          [ ] decline",
        "Analyst: ____________",
    ])


# --------------------------------------------------------------------------------------
# Section 1: Single application
# --------------------------------------------------------------------------------------
PRESETS = {
    "Typical green case": 1601,
    "Typical yellow case": 1759,
    "Typical red case": 2382,
    "Older good customer (age 56)": 4322,
}
GROUPS = {
    "Loan": ["LoanAmount", "LoanDuration", "LoanPurpose", "InstallmentPercent",
             "InstallmentPlans", "OthersOnLoan"],
    "Finances and credit history": ["CheckingStatus", "ExistingSavings", "CreditHistory",
                                    "ExistingCreditsCount", "OwnsProperty"],
    "Personal and employment": ["Age", "EmploymentDuration", "Job", "Housing",
                                "CurrentResidenceDuration", "Dependents", "Telephone"],
}


def load_preset():
    name = st.session_state["preset"]
    if name in PRESETS:
        row = dataset.loc[PRESETS[name]]
        for f in INPUT_FEATURES:
            st.session_state[f"f_{f}"] = row[f].item() if hasattr(row[f], "item") else row[f]


def feature_input(f):
    key = f"f_{f}"
    if f in CATEGORICAL:
        options = META["category_values"][f]
        st.selectbox(feature_label(f), options, key=key, format_func=value_label)
    else:
        lo, hi = META["numeric_ranges"][f]
        if f == "LoanAmount":
            st.number_input("Loan amount", min_value=lo, max_value=hi, step=100, key=key)
        else:
            st.slider("Age" if f == "Age" else feature_label(f), lo, hi, key=key)


def section_single():
    st.subheader("Single application")
    st.markdown('<span class="cp-muted">Enter an application or load an example. '
                'The result updates immediately.</span>', unsafe_allow_html=True)
    if "f_Age" not in st.session_state:
        st.session_state["preset"] = "Typical yellow case"
        load_preset()

    left, right = st.columns([1, 1.25], gap="large")
    with left:
        st.selectbox("Load an example from the test set", list(PRESETS) + ["Custom"],
                     key="preset", on_change=load_preset)
        for title, feats in GROUPS.items():
            with st.expander(title, expanded=True):
                cols = st.columns(2)
                for i, f in enumerate(feats):
                    with cols[i % 2]:
                        feature_input(f)
        with st.expander("Not used for scoring", expanded=False):
            st.caption("Recorded with the application, but not an input of the final model: "
                       "no applicant's score depends directly on sex or foreign worker status.")
            cols = st.columns(2)
            for i, f in enumerate(NOT_SCORED):
                with cols[i]:
                    feature_input(f)

    row = pd.DataFrame([{f: st.session_state[f"f_{f}"] for f in INPUT_FEATURES}])
    p = float(predict(row, "d")[0])
    lower, upper, target = current_policy("d")
    zone = zone_of(p, lower, upper).item()
    rating = rating_of(p)
    contrib, _ = explain(row, "d")
    contrib = contrib.iloc[0]

    with right:
        with st.container(border=True):
            st.markdown('<span class="cp-muted">Default probability</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="cp-pd">{p:.1%}</div>', unsafe_allow_html=True)
            st.markdown(zone_badge(zone, rating), unsafe_allow_html=True)
            st.markdown(f'<p class="cp-muted" style="margin-top:.6rem">{ZONES[zone]["text"]}</p>',
                        unsafe_allow_html=True)
            st.caption(f"Rating = fixed risk scale (A < 5% … E ≥ 70%). Zone = bank policy: "
                       f"green below {lower:.2f}, red from {upper:.2f} (target precision {target:.0%}; "
                       f"adjustable in the Bank view).")

        st.markdown("**Top drivers of this decision**")
        st.altair_chart(drivers_chart(contrib, row.iloc[0]), width="stretch")
        st.caption("Exact SHAP values of the logistic regression, relative to the average applicant. "
                   "Age is shown as a fairness correction: it offsets age information carried by other "
                   "features. Explanations are for analysts; customer-facing explanations are a next step.")

        if zone == "Yellow":
            st.markdown("**Credit memo for the analyst**")
            memo = credit_memo(row.iloc[0], p, rating, zone, contrib, lower, upper)
            st.code(memo, language=None, wrap_lines=True)
            st.download_button("Download credit memo", memo, file_name="credit_memo.txt")

        with st.expander("Compare with model (a) – all features, no fairness correction"):
            pa = float(predict(row, "a")[0])
            lo_a, up_a, _ = current_policy("a")
            za = zone_of(pa, lo_a, up_a).item()
            st.markdown(f"Model (a): default probability **{pa:.1%}** · " + zone_badge(za, rating_of(pa)),
                        unsafe_allow_html=True)
            st.caption("Model (a) uses sex and foreign worker status and no fairness correction "
                       f"(boundaries {lo_a:.2f} / {up_a:.2f}).")


# --------------------------------------------------------------------------------------
# Section 2: Batch upload
# --------------------------------------------------------------------------------------
@st.cache_data
def sample_csv():
    # 25 test applications spread evenly over the range of default probabilities
    test = evaluation[evaluation["split"] == "test"].sort_values("pd_final")
    test_ids = test.index[np.linspace(0, len(test) - 1, 25).astype(int)]
    return dataset.loc[test_ids, INPUT_FEATURES].to_csv(index=False).encode()


def section_batch():
    st.subheader("Batch upload")
    st.markdown('<span class="cp-muted">Upload a CSV with one application per row and the '
                f'{len(INPUT_FEATURES)} application fields as columns.</span>', unsafe_allow_html=True)
    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader("CSV file", type="csv")
    with c2:
        st.download_button("Download sample CSV (25 applications)", sample_csv(),
                           file_name="sample_applications.csv", width="stretch")
        use_sample = st.toggle("Use the sample file", value=uploaded is None, key="use_sample")

    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif use_sample:
        data = pd.read_csv(pd.io.common.BytesIO(sample_csv()))
    else:
        st.info("Upload a CSV file or use the sample file.")
        return

    missing = [f for f in INPUT_FEATURES if f not in data.columns]
    if missing:
        st.error("Missing columns: " + ", ".join(missing))
        return
    data = data.drop(columns=[c for c in ["Risk"] if c in data.columns])

    results = score_table(data)
    lower, upper, target = current_policy("d")
    zones = zone_of(results["Default probability (%)"].to_numpy() / 100, lower, upper)
    cols = st.columns(4)
    cols[0].metric("Applications", f"{len(results):,}")
    for col, z in zip(cols[1:], ZONE_ORDER):
        col.metric(f"{ZONES[z]['icon']} {ZONES[z]['label']}",
                   f"{(zones == z).sum():,} ({(zones == z).mean():.0%})")

    st.dataframe(
        results, hide_index=True, width="stretch", height=420,
        column_config={"Default probability (%)": st.column_config.ProgressColumn(
            "Default probability", format="%.1f%%", min_value=0, max_value=100)},
    )
    st.caption(f"Zones use the current policy: green below {lower:.2f}, red from {upper:.2f} "
               f"(target precision {target:.0%}). Sex and foreign worker status are not used for scoring.")
    st.download_button("Download results as CSV", results.to_csv(index=False).encode(),
                       file_name="creditparity_results.csv", type="primary")


# --------------------------------------------------------------------------------------
# Section 3: Bank view
# --------------------------------------------------------------------------------------
def policy_metrics(model_key, lower, upper):
    test = evaluation[evaluation["split"] == "test"]
    p = test["pd_final" if model_key == "d" else "pd_comparison"].to_numpy()
    y = test["y"].to_numpy()
    zone = zone_of(p, lower, upper)
    decline = p >= lower
    good51 = (test["age_group"].to_numpy() == "51+") & (y == 0)
    return {
        "Auto-approve": (zone == "Green").mean(),
        "Manual review": (zone == "Yellow").mean(),
        "Decline recommended": (zone == "Red").mean(),
        "Cost": int(5 * ((~decline) & (y == 1)).sum() + ((decline) & (y == 0)).sum()),
        "Recall": ((decline) & (y == 1)).sum() / (y == 1).sum(),
        "Precision": y[decline].mean() if decline.any() else np.nan,
        "Red-zone precision": y[zone == "Red"].mean() if (zone == "Red").any() else np.nan,
        "Good customers auto-declined": int(((zone == "Red") & (y == 0)).sum()),
        "Defaults auto-approved": int(((zone == "Green") & (y == 1)).sum()),
        "Good 51+ auto-declined": f"{int((good51 & (zone == 'Red')).sum())} of {int(good51.sum())}",
    }


def fairness_tables(model_key, lower, upper):
    test = evaluation[evaluation["split"] == "test"].copy()
    p = test["pd_final" if model_key == "d" else "pd_comparison"].to_numpy()
    test["decline"] = p >= lower
    test["zone"] = zone_of(p, lower, upper)
    groups = {"Sex": "Sex", "Age group": "age_group", "Foreign worker": "ForeignWorker"}
    per_group, summary = [], []
    for name, col in groups.items():
        rows = []
        for g, d in test.groupby(col):
            good, bad = d[d["y"] == 0], d[d["y"] == 1]
            rows.append({"Attribute": name, "Group": g, "Applications": len(d),
                         "Rejection rate": d["decline"].mean(),
                         "Good customers declined (FPR)": good["decline"].mean(),
                         "Defaults caught (recall)": bad["decline"].mean(),
                         "Green": (d["zone"] == "Green").mean(), "Yellow": (d["zone"] == "Yellow").mean(),
                         "Red": (d["zone"] == "Red").mean()})
        rows = pd.DataFrame(rows)
        per_group.append(rows)
        summary.append({"Attribute": name,
                        "Demographic parity difference": rows["Rejection rate"].max() - rows["Rejection rate"].min(),
                        "Equal opportunity difference": rows["Good customers declined (FPR)"].max()
                        - rows["Good customers declined (FPR)"].min()})
    return pd.concat(per_group, ignore_index=True), pd.DataFrame(summary)


def zone_share_chart(per_group):
    rows = []
    for _, r in per_group.iterrows():
        label = f"Foreign worker: {r['Group']}" if r["Attribute"] == "Foreign worker" else str(r["Group"])
        start = 0.0
        for z in ZONE_ORDER:
            rows.append({"Group": label, "Zone": f"{ZONES[z]['icon']} {z} – {ZONES[z]['label']}",
                         "start": start, "end": start + r[z], "Share (%)": round(100 * r[z], 1)})
            start += r[z]
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df["Group"]))
    zone_labels = [f"{ZONES[z]['icon']} {z} – {ZONES[z]['label']}" for z in ZONE_ORDER]
    return alt.Chart(df).mark_bar(stroke=SURFACE, strokeWidth=2).encode(
        x=alt.X("start:Q", title="Share of applications", scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", gridColor="#ececea", labelColor=TEXT_SECONDARY, titleColor=TEXT_SECONDARY)),
        x2="end:Q",
        y=alt.Y("Group:N", sort=order, title=None, scale=alt.Scale(paddingInner=0.25),
                axis=alt.Axis(labelColor="#0b0b0b", labelLimit=200)),
        color=alt.Color("Zone:N", sort=zone_labels, scale=alt.Scale(domain=zone_labels,
                        range=[ZONES[z]["color"] for z in ZONE_ORDER]),
                        legend=alt.Legend(orient="top", title=None, labelLimit=260)),
        tooltip=["Group", "Zone", "Share (%)"],
    ).properties(height=48 * len(order)).configure_view(stroke=None)


def workload_chart(lower, target):
    targets = np.round(np.arange(0.70, 0.951, 0.01), 2)
    test = evaluation[evaluation["split"] == "test"]
    p = test["pd_final"].to_numpy()
    rows = [{"Target precision": t,
             "Manual review": ((p >= lower) & (p < upper_for_precision("d", lower, float(t)))).mean()}
            for t in targets]
    df = pd.DataFrame(rows)
    df["Manual review (%)"] = (100 * df["Manual review"]).round(1)
    base = alt.Chart(df).encode(
        x=alt.X("Target precision:Q", axis=alt.Axis(format="%", labelColor=TEXT_SECONDARY, titleColor=TEXT_SECONDARY),
                scale=alt.Scale(domain=[0.70, 0.95])),
        y=alt.Y("Manual review:Q", title="Manual review",
                axis=alt.Axis(format="%", gridColor="#ececea", labelColor=TEXT_SECONDARY, titleColor=TEXT_SECONDARY),
                scale=alt.Scale(zero=False)),
    )
    line = base.mark_line(color=MODEL_COLORS["d"], strokeWidth=2)
    hover = base.mark_circle(size=120, opacity=0).encode(tooltip=["Target precision", "Manual review (%)"])
    current = df[np.isclose(df["Target precision"], round(target, 2))]
    point = alt.Chart(current).mark_circle(size=110, color=MODEL_COLORS["d"], stroke=SURFACE, strokeWidth=2).encode(
        x="Target precision:Q", y="Manual review:Q", tooltip=["Target precision", "Manual review (%)"])
    return (line + hover + point).properties(height=230).configure_view(stroke=None)


def section_bank():
    st.subheader("Bank view – decision policy")
    st.markdown('<span class="cp-muted">Move the sliders to see how the policy changes analyst workload, '
                'cost and fairness. All figures are computed on the 1,000 test applications.</span>',
                unsafe_allow_html=True)

    controls, kpis = st.columns([1, 2.2], gap="large")
    with controls:
        with st.container(border=True):
            st.markdown("**Policy for the final model (d)**")
            st.slider("Lower boundary: auto-approve below", 0.05, 0.40, step=0.01,
                      value=st.session_state["lower_d"], key="w_lower_d",
                      on_change=persist, args=("w_lower_d", "lower_d"),
                      help=f"Cost-optimal value (5:1 cost matrix): {POLICY['models']['d']['lower_boundary']:.2f}")
            st.slider("Target precision for automatic declines", 0.70, 0.95, step=0.01,
                      value=st.session_state["target"], key="w_target", format="%.2f",
                      on_change=persist, args=("w_target", "target"),
                      help="Share of red-zone applications that must be actual defaults. "
                           "The upper boundary is the lowest threshold that meets this target.")
            lower, upper, target = current_policy("d")
            m = policy_metrics("d", lower, upper)
            st.markdown(f"Upper boundary: **{upper:.2f}** → manual review: **{m['Manual review']:.1%}** "
                        "of applications")
            st.button("Reset to the approved policy", on_click=reset_policy)
            st.toggle("Compare with model (a)", key="compare",
                      help="Model (a): all features incl. sex, no fairness correction; its own "
                           "cost-optimal lower boundary and the same target precision.")

    compare = st.session_state["compare"]
    baseline_cost = policy_metrics("a", 0.5, 0.5)["Cost"]          # (a) at the default 0.5 threshold
    with kpis:
        lo_a, up_a, _ = current_policy("a")
        ma = policy_metrics("a", lo_a, up_a)
        r1 = st.columns(3)
        for col, key, z in zip(r1, ["Auto-approve", "Manual review", "Decline recommended"], ZONE_ORDER):
            delta = f"(a): {ma[key]:.1%}" if compare else None
            col.metric(f"{ZONES[z]['icon']} {key}", f"{m[key]:.1%}", delta, delta_color="off", delta_arrow="off")
        r2 = st.columns(4)
        r2[0].metric("Cost (5·FN + 1·FP)", f"{m['Cost']:,}",
                     f"{m['Cost'] / baseline_cost - 1:+.0%} vs. 0.5", delta_color="inverse",
                     help="Compared with model (a) at the default threshold 0.5")
        r2[1].metric("Recall (defaults caught)", f"{m['Recall']:.1%}",
                     f"(a): {ma['Recall']:.1%}" if compare else None, delta_color="off", delta_arrow="off")
        r2[2].metric("Precision of declines", f"{m['Precision']:.1%}",
                     f"(a): {ma['Precision']:.1%}" if compare else None, delta_color="off", delta_arrow="off")
        r2[3].metric("Red-zone precision", f"{m['Red-zone precision']:.1%}",
                     f"(a): {ma['Red-zone precision']:.1%}" if compare else None, delta_color="off", delta_arrow="off")
        r3 = st.columns(3)
        r3[0].metric("Good customers auto-declined", m["Good customers auto-declined"],
                     f"(a): {ma['Good customers auto-declined']}" if compare else None, delta_color="off", delta_arrow="off")
        r3[1].metric("Defaults auto-approved", m["Defaults auto-approved"],
                     f"(a): {ma['Defaults auto-approved']}" if compare else None, delta_color="off", delta_arrow="off")
        r3[2].metric("Good customers 51+ in red", m["Good 51+ auto-declined"],
                     f"(a): {ma['Good 51+ auto-declined']}" if compare else None, delta_color="off", delta_arrow="off")
        st.caption(f"Cost counts every application with a default probability ≥ the lower boundary as declined "
                   f"(single-threshold logic). Baseline: model (a) at the default threshold 0.5 = {baseline_cost} "
                   + (f"· Model (a) boundaries: {lo_a:.2f} / {up_a:.2f}, cost {ma['Cost']}." if compare else ""))

    st.markdown("#### Analyst workload vs. target precision")
    st.altair_chart(workload_chart(lower, target), width="stretch")
    st.caption("A higher target means fewer wrongful automatic declines, but more cases for the analysts.")

    st.markdown("#### Fairness")
    per_group, summary = fairness_tables("d", lower, upper)
    shown = [("(d) Final model", summary)]
    if compare:
        pg_a, sum_a = fairness_tables("a", lo_a, up_a)
        shown.append(("(a) Comparison model", sum_a))
    cols = st.columns(len(shown))
    for col, (name, s) in zip(cols, shown):
        with col:
            st.markdown(f"**{name}**")
            st.dataframe(s.set_index("Attribute").style.format("{:.2f}"), width="stretch")
    st.caption("Demographic parity difference = largest gap in rejection rates between groups. "
               "Equal opportunity difference = largest gap in the share of good customers who are declined. "
               "0 = no gap. Rejection = default probability ≥ lower boundary.")

    st.markdown("**Zone shares by group – final model (d)**")
    st.altair_chart(zone_share_chart(per_group), width="stretch")
    st.markdown("**Metrics by group – final model (d)**")
    table = per_group.set_index(["Attribute", "Group"])[
        ["Applications", "Rejection rate", "Good customers declined (FPR)", "Defaults caught (recall)"]]
    st.dataframe(table.style.format({"Rejection rate": "{:.0%}", "Good customers declined (FPR)": "{:.0%}",
                                     "Defaults caught (recall)": "{:.0%}"}), width="stretch")
    if compare:
        st.markdown("**Zone shares by group – comparison model (a)**")
        st.altair_chart(zone_share_chart(pg_a), width="stretch")
    st.caption("Small groups: only 23 test applicants are not foreign workers and 21 good customers are "
               "aged 51+, so their figures are uncertain.")


# --------------------------------------------------------------------------------------
# Section 4: Model info
# --------------------------------------------------------------------------------------
def section_info():
    st.subheader("Model info")

    st.markdown("#### Rating and zone")
    st.markdown("**The rating (A–E) describes the applicant's risk and depends only on the default "
                "probability; the zone (green / yellow / red) describes how the bank handles the "
                "application and follows the policy settings, so moving a slider can change the zone "
                "but never the rating.**")
    scale = pd.DataFrame([{"Rating": r, "Default probability": f"{lo:.0%} – {min(hi, 1):.0%}"
                           if hi <= 1 else f"≥ {lo:.0%}"} for r, lo, hi in RATING_SCALE])
    lower, upper, target = current_policy("d")
    zones = pd.DataFrame([
        {"Zone": f"{ZONES['Green']['icon']} Green – auto-approve", "Rule": f"PD < {lower:.2f}"},
        {"Zone": f"{ZONES['Yellow']['icon']} Yellow – manual review", "Rule": f"{lower:.2f} ≤ PD < {upper:.2f}"},
        {"Zone": f"{ZONES['Red']['icon']} Red – decline recommended", "Rule": f"PD ≥ {upper:.2f}"},
    ])
    c1, c2 = st.columns(2)
    c1.dataframe(scale, hide_index=True, width="stretch")
    c2.dataframe(zones, hide_index=True, width="stretch")
    st.caption("Lower boundary = cost-optimal threshold for a 5:1 cost ratio (missed default vs. rejected good "
               "customer). Upper boundary = lowest threshold at which at least 80% of automatic declines are "
               "actual defaults. Both chosen with 5-fold cross-validation on the training data.")

    st.markdown("#### Why a simple model?")
    cv = pd.DataFrame([
        ("Logistic regression", 0.857, 0.010, "✔ selected"),
        ("XGBoost (tuned)", 0.850, 0.009, ""),
        ("Random forest (tuned)", 0.848, 0.009, ""),
        ("Random forest (default)", 0.844, 0.007, ""),
        ("XGBoost (default)", 0.829, 0.010, ""),
    ], columns=["Model", "ROC AUC (5-fold CV)", "Std", ""])
    c1, c2 = st.columns([1.1, 1], gap="large")
    with c1:
        st.dataframe(cv, hide_index=True, width="stretch",
                     column_config={"ROC AUC (5-fold CV)": st.column_config.NumberColumn(format="%.3f"),
                                    "Std": st.column_config.NumberColumn(format="%.3f")})
    with c2:
        st.markdown(
            "- The logistic regression has the best ROC AUC and wins **in all 5 folds** against every tree model.\n"
            "- An initial analysis with **IBM watsonx AutoAI** selected an XGBoost pipeline by accuracy "
            "(79.6% on a 499-case holdout), but it caught only **53% of defaults**.\n"
            "- The real gain comes from the **cost-based threshold**: recall rises from 63% to 95%.\n"
            "- A logistic regression is transparent: every driver of a decision can be shown exactly, which "
            "matters for a high-risk AI system under the EU AI Act.")

    st.markdown("#### The price of fairness")
    lo_a, up_a, _ = current_policy("a")
    md_, ma = policy_metrics("d", lower, upper), policy_metrics("a", lo_a, up_a)
    _, sd = fairness_tables("d", lower, upper)
    _, sa = fairness_tables("a", lo_a, up_a)
    eod = lambda s, a: s.set_index("Attribute").loc[a, "Equal opportunity difference"]
    comp = pd.DataFrame({
        "(d) Fair model – final": [f"{lower:.2f} / {upper:.2f}", md_["Cost"], f"{md_['Recall']:.1%}",
                                   f"{md_['Manual review']:.0%}", md_["Defaults auto-approved"],
                                   md_["Good 51+ auto-declined"], f"{eod(sd, 'Age group'):.2f}",
                                   f"{eod(sd, 'Sex'):.2f}", "No"],
        "(a) All features – comparison": [f"{lo_a:.2f} / {up_a:.2f}", ma["Cost"], f"{ma['Recall']:.1%}",
                                          f"{ma['Manual review']:.0%}", ma["Defaults auto-approved"],
                                          ma["Good 51+ auto-declined"], f"{eod(sa, 'Age group'):.2f}",
                                          f"{eod(sa, 'Sex'):.2f}", "Yes"],
    }, index=["Boundaries", "Cost (test set)", "Recall", "Manual review", "Defaults auto-approved",
              "Good customers 51+ auto-declined", "Age gap (EOD)", "Sex gap (EOD)", "Uses sex as input"])
    st.dataframe(comp.astype(str), width="stretch")
    st.markdown(
        "- **Final model (d):** logistic regression trained with *reweighing* on age group, sex and foreign "
        "worker status. **Sex and foreign worker status are not inputs**: two applicants who differ only in "
        "sex get the same score. Age stays as an input to correct for the many features that reveal age "
        "indirectly (dropping it does not remove the age gap).\n"
        "- **Equal treatment vs. equal outcomes.** A tested alternative that uses sex as an input almost "
        "removes the sex gap, but only by rating women as riskier than men with otherwise identical features. "
        "Every model that treats men and women equally shows a gap of about 0.2 against men on this data. "
        "CreditParity chooses equal treatment.")

    st.markdown("#### Limitations")
    st.markdown(
        "- The dataset is **synthetic** (IBM version of German Credit). Some patterns are implausible, e.g. a "
        "negative checking balance lowers the predicted risk.\n"
        "- The **5:1 cost ratio** and the **80% precision target** are business assumptions. Before deployment "
        "they should be validated with the bank's risk and compliance teams; thresholds should be monitored "
        "and adjusted to the available analyst capacity.\n"
        "- About **60% of applications go to manual review** with the approved policy.\n"
        "- The remaining **sex gap of about 0.2** works against men and results from differences in other "
        "features. It needs legal review (direct vs. indirect discrimination) and monitoring.\n"
        "- **Age is used directly** as a fairness correction, and reweighing uses age, sex and foreign worker "
        "status to compute training weights. Both need legal review.\n"
        "- The **age correction is strong**: all else equal, ten years younger multiplies the odds of default by "
        "2.55, so age alone can move an application across zones. It **shifts predicted risk toward younger "
        "applicants** (average predicted default probability 21% for ages 19–25 vs. an observed default rate of "
        "5%; 50% vs. 76% for ages 51+).\n"
        "- Reweighing declines more young good customers than the comparison model; most of them go to "
        "manual review.\n"
        "- Explanations are **analyst-facing only**. A customer-facing explanation (GDPR Art. 22, right to "
        "meaningful information) is a next step that requires legal review.\n"
        "- Cost figures assume that every application with a default probability above the lower boundary is "
        "declined; analyst decisions may change this.\n"
        "- The models are trained on 4,000 applications so that the test results stay verifiable; a production "
        "model would be refit on all data and revalidated.")
    st.caption(f"Model trained with scikit-learn {META['sklearn_version']}. Final team project, "
               "BUS/FIN 241A Machine Learning, Brandeis University.")


# --------------------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------------------
SECTIONS = {"single": "1 · Single application", "batch": "2 · Batch upload",
            "bank": "3 · Bank view", "info": "4 · Model info"}


def main():
    init_state()
    header()
    requested = st.query_params.get("section", "single")
    keys = list(SECTIONS)
    with st.sidebar:
        st.markdown("### 🏦 CreditParity")
        choice = st.radio("Section", keys, index=keys.index(requested) if requested in keys else 0,
                          format_func=SECTIONS.get, label_visibility="collapsed")
        st.query_params["section"] = choice
        st.divider()
        lower, upper, target = current_policy("d")
        st.markdown("**Active policy – model (d)**")
        st.markdown(f"{ZONES['Green']['icon']} Green: PD < {lower:.2f}  \n"
                    f"{ZONES['Yellow']['icon']} Yellow: {lower:.2f} – {upper:.2f}  \n"
                    f"{ZONES['Red']['icon']} Red: PD ≥ {upper:.2f}")
        st.caption(f"Target precision for automatic declines: {target:.0%}. Change it in the Bank view.")
        st.divider()
        st.caption("Analyst tool · explanations are for internal use. Sex and foreign worker status are "
                   "not used for scoring.")

    {"single": section_single, "batch": section_batch,
     "bank": section_bank, "info": section_info}[choice]()


main()
