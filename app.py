"""CreditParity – Profitable, Fair Lending Decisions.

Streamlit app for bank staff. Sections:
  Home                        – mission and overview
  Check a single application  – chance of default, risk rating, recommendation and main reasons
  Check several applications  – score a CSV file and download the results
  Bank settings               – decision rules and their effect on workload, cost of wrong decisions and fairness
  How it works                – plain-language explanation and important notes

Models and data are loaded from model/ (created in notebook Step 8). Technical details are
documented in the notebooks and the README of the repository.
"""
from pathlib import Path
import base64

import altair as alt
import joblib
import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).parent
MODEL_DIR = ROOT / "model"
DATA_PATH = ROOT / "data" / "german_credit_data_biased_training.csv"
LOGO_PATH = ROOT / "assets" / "logo.svg"

st.set_page_config(page_title="CreditParity", page_icon=str(LOGO_PATH), layout="wide",
                   initial_sidebar_state="expanded")

# --------------------------------------------------------------------------------------
# Design tokens
# --------------------------------------------------------------------------------------
NAVY = "#0d366b"
TEXT_SECONDARY = "#52514e"
SURFACE = "#fcfcfb"
RAISES_RISK, LOWERS_RISK = "#e34948", "#2a78d6"            # diverging pair (red / blue)
ZONES = {                                                  # status palette + icon + label
    "Green": {"label": "Approve automatically", "short": "Approved automatically", "color": "#0ca30c",
              "icon": "✅", "advice": "Low risk. The application can be approved automatically."},
    "Yellow": {"label": "Manual review", "short": "Reviewed by an analyst", "color": "#fab219",
               "icon": "🔎", "advice": "Uncertain case. A credit analyst should review the application "
                                      "and make the decision."},
    "Red": {"label": "Decline recommended", "short": "Declined automatically", "color": "#d03b3b",
            "icon": "⛔", "advice": "High risk. The application is declined with an explanation. "
                                   "The applicant can ask for a personal review."},
}
ZONE_ORDER = ["Green", "Yellow", "Red"]

# Fixed risk scale: the rating describes the applicant's risk and never moves with the bank settings.
RATING_SCALE = [("A", 0.00, 0.05), ("B", 0.05, 0.15), ("C", 0.15, 0.35),
                ("D", 0.35, 0.70), ("E", 0.70, 1.01)]
RATING_TEXT = {"A": "very low risk", "B": "low risk", "C": "moderate risk",
               "D": "elevated risk", "E": "high risk"}

# Plain-language names for the application fields and their values
FIELD_LABELS = {
    "LoanAmount": "Loan amount",
    "LoanDuration": "Loan duration (months)",
    "LoanPurpose": "Purpose of the loan",
    "InstallmentPercent": "Monthly installment burden (1 = low, 6 = high)",
    "InstallmentPlans": "Other installment plans",
    "OthersOnLoan": "Co-applicant or guarantor",
    "CheckingStatus": "Checking account balance",
    "ExistingSavings": "Savings",
    "CreditHistory": "Credit history",
    "ExistingCreditsCount": "Number of existing loans",
    "OwnsProperty": "Most valuable property",
    "Age": "Age",
    "EmploymentDuration": "Time with current employer",
    "Job": "Type of job",
    "Housing": "Housing",
    "CurrentResidenceDuration": "Years at current address (1–6)",
    "Dependents": "Number of dependents",
    "Telephone": "Telephone registered",
}
VALUE_LABELS = {
    "CheckingStatus": {"less_0": "Overdrawn (below 0)", "0_to_200": "0 to 200",
                       "greater_200": "More than 200", "no_checking": "No checking account"},
    "CreditHistory": {"all_credits_paid_back": "All previous loans repaid",
                      "credits_paid_to_date": "Current loans paid on time",
                      "prior_payments_delayed": "Payments delayed in the past",
                      "outstanding_credit": "Other loans outstanding", "no_credits": "No previous loans"},
    "LoanPurpose": {"car_new": "New car", "car_used": "Used car", "furniture": "Furniture",
                    "appliances": "Household appliances", "radio_tv": "Radio / TV", "repairs": "Repairs",
                    "education": "Education", "retraining": "Retraining", "business": "Business",
                    "vacation": "Vacation", "other": "Other"},
    "ExistingSavings": {"less_100": "Less than 100", "100_to_500": "100 to 500", "500_to_1000": "500 to 1,000",
                        "greater_1000": "More than 1,000", "unknown": "Unknown / no savings"},
    "EmploymentDuration": {"unemployed": "Unemployed", "less_1": "Less than 1 year", "1_to_4": "1 to 4 years",
                           "4_to_7": "4 to 7 years", "greater_7": "More than 7 years"},
    "OthersOnLoan": {"none": "None", "co-applicant": "Co-applicant", "guarantor": "Guarantor"},
    "OwnsProperty": {"real_estate": "Real estate", "savings_insurance": "Savings or life insurance",
                     "car_other": "Car or other", "unknown": "None / unknown"},
    "InstallmentPlans": {"none": "None", "bank": "At a bank", "stores": "At stores"},
    "Housing": {"own": "Owns home", "rent": "Rents", "free": "Lives rent-free"},
    "Job": {"unemployed": "Unemployed", "unskilled": "Unskilled", "skilled": "Skilled employee",
            "management_self-employed": "Management / self-employed"},
    "Telephone": {"yes": "Yes", "none": "No"},
}
FORM_GROUPS = {
    "The loan": ["LoanAmount", "LoanDuration", "LoanPurpose", "InstallmentPercent",
                 "InstallmentPlans", "OthersOnLoan"],
    "Finances and credit history": ["CheckingStatus", "ExistingSavings", "CreditHistory",
                                    "ExistingCreditsCount", "OwnsProperty"],
    "Personal situation": ["Age", "EmploymentDuration", "Job", "Housing",
                           "CurrentResidenceDuration", "Dependents", "Telephone"],
}
NOT_USED = ["Sex", "ForeignWorker"]                         # not used by the final model
AGE_LABEL = "Age (fairness correction)"

st.markdown(f"""
<style>
  .block-container {{padding-top: 3.2rem; max-width: 1200px;}}
  h1, h2, h3, h4 {{color: {NAVY};}}
  .cp-header {{display:flex; align-items:center; gap:.7rem; margin-bottom:1.2rem;}}
  .cp-title {{font-size:1.5rem; font-weight:700; color:{NAVY}; letter-spacing:-.01em;}}
  .cp-tagline {{font-size:1rem; color:{TEXT_SECONDARY};}}
  /* centered page title + intro, used on every page */
  .cp-intro {{text-align:center; max-width:720px; margin:0 auto 1.6rem auto;}}
  .cp-page-title {{font-size:1.9rem; font-weight:700; color:{NAVY}; line-height:1.25; margin:0 0 .45rem 0;}}
  .cp-intro p {{font-size:1.05rem; line-height:1.6; color:{TEXT_SECONDARY}; margin:0;}}
  .cp-section {{font-size:1.25rem; font-weight:700; color:{NAVY}; margin:1.8rem 0 .8rem 0;}}
  .cp-section.cp-center {{text-align:center;}}
  .cp-badge {{display:inline-flex; align-items:center; gap:.45rem; padding:.3rem .8rem;
             border-radius:999px; font-weight:600; font-size:1rem; border:1px solid #d9dde3;
             background:#ffffff; color:#0b0b0b;}}
  .cp-dot {{width:.75rem; height:.75rem; border-radius:50%; display:inline-block;}}
  .cp-pd {{font-size:3.2rem; font-weight:700; color:{NAVY}; line-height:1.1;}}
  .cp-muted {{color:{TEXT_SECONDARY}; font-size:.95rem;}}
  .cp-hero {{text-align:center; padding:2rem 1rem 0 1rem; max-width:720px; margin:0 auto;}}
  .cp-hero img {{display:block; margin:0 auto;}}
  .cp-hero-title {{font-size:3rem; font-weight:700; color:{NAVY}; line-height:1.15; margin:.8rem 0 .3rem 0;}}
  .cp-hero .cp-tagline {{font-size:1.25rem; margin:0;}}
  .cp-hero p.cp-mission {{margin:1.3rem 0 0 0 !important; font-size:1.1rem; line-height:1.65; color:#0b0b0b;}}
  /* card rows: CSS grid, so every card in a row has the same height */
  .cp-grid {{display:grid; gap:1rem; align-items:stretch; margin-bottom:.5rem;}}
  .cp-grid-4 {{grid-template-columns:repeat(4, minmax(0, 1fr));}}
  .cp-grid-3 {{grid-template-columns:repeat(3, minmax(0, 1fr));}}
  @media (max-width: 900px) {{ .cp-grid-4, .cp-grid-3 {{grid-template-columns:1fr;}} }}
  .cp-card {{background:#ffffff; border:1px solid #e3e6ea; border-radius:12px; padding:1.1rem 1.2rem;
            display:flex; flex-direction:column;}}
  .cp-card-title {{font-size:1.05rem; font-weight:700; color:{NAVY}; margin:0 0 .45rem 0; line-height:1.35;}}
  .cp-card p {{margin:0; color:{TEXT_SECONDARY}; font-size:.95rem; line-height:1.5;}}
  .cp-author {{text-align:center; color:{TEXT_SECONDARY}; margin-top:2.2rem; font-size:1rem;}}
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


@st.cache_data
def logo_data_uri():
    return "data:image/svg+xml;base64," + base64.b64encode(LOGO_PATH.read_bytes()).decode()


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
# Helpers: labels, rating, zones, settings
# --------------------------------------------------------------------------------------
def field_label(feature):
    return AGE_LABEL if feature == "Age" else FIELD_LABELS.get(feature, feature)


def value_label(feature, value):
    if feature in VALUE_LABELS:
        return VALUE_LABELS[feature].get(value, str(value))
    return f"{value:,}" if isinstance(value, (int, np.integer)) else str(value)


def rating_of(p):
    return next(r for r, lo, hi in RATING_SCALE if lo <= p < hi)


def zone_of(p, lower, upper):
    p = np.asarray(p)
    return np.where(p < lower, "Green", np.where(p >= upper, "Red", "Yellow"))


THRESHOLDS = np.round(np.arange(0.01, 1.00, 0.01), 2)


@st.cache_data
def upper_for_precision(model_key, lower, target):
    """Lowest threshold >= lower at which at least `target` of the automatic declines are
    actual defaults (on cross-validated predictions of the training applications)."""
    oof = evaluation[evaluation["split"] == "train_oof"]
    p = oof["pd_final" if model_key == "d" else "pd_comparison"].to_numpy()
    y = oof["y"].to_numpy()
    for t in THRESHOLDS[THRESHOLDS >= lower]:
        red = p >= t
        if red.any() and y[red].mean() >= target:
            return float(t)
    return 1.0


@st.cache_data
def form_defaults():
    """Start values for the form: the most common value (or the median) of each field."""
    out = {}
    for f in INPUT_FEATURES:
        v = dataset[f].mode().iloc[0] if f in CATEGORICAL else int(dataset[f].median())
        out[f] = v.item() if hasattr(v, "item") else v
    return out


def init_state():
    # Keep form values when the user switches sections (Streamlit drops the state of
    # widgets that are not shown in a run).
    for k in list(st.session_state.keys()):
        if k.startswith("f_"):
            st.session_state[k] = st.session_state[k]
    st.session_state.setdefault("lower_d", POLICY["models"]["d"]["lower_boundary"])
    st.session_state.setdefault("target", POLICY["target_precision"])
    for f, v in form_defaults().items():
        if f not in NOT_USED:
            st.session_state.setdefault(f"f_{f}", v)


# The sliders work in whole percent (15 = 15%); the settings are stored as shares (0.15).
SETTING_WIDGETS = {"lower_d": "w_lower_d", "target": "w_target"}


def to_percent(share):
    return int(round(share * 100))


def persist(state_key):
    st.session_state[state_key] = round(st.session_state[SETTING_WIDGETS[state_key]] / 100, 2)


def reset_settings():
    recommended = {"lower_d": POLICY["models"]["d"]["lower_boundary"], "target": POLICY["target_precision"]}
    for state_key, value in recommended.items():
        st.session_state[state_key] = value
        st.session_state[SETTING_WIDGETS[state_key]] = to_percent(value)


def current_policy(model_key="d"):
    lower = (st.session_state["lower_d"] if model_key == "d"
             else POLICY["models"]["a"]["lower_boundary"])
    target = st.session_state["target"]
    return lower, upper_for_precision(model_key, lower, target), target


# --------------------------------------------------------------------------------------
# Scoring and explanations
# --------------------------------------------------------------------------------------
def complete(data):
    """Add the fields the final model does not use if they are missing (the scoring pipeline
    expects all application fields, but ignores these two)."""
    data = data.copy()
    defaults = form_defaults()
    for f in NOT_USED:
        if f not in data.columns:
            data[f] = defaults[f]
    return data[INPUT_FEATURES]


def predict(data, model_key="d"):
    return MODELS[model_key].predict_proba(complete(data))[:, 1]


def explain(data, model_key="d"):
    """Exact contribution of each application field to the risk (SHAP values of the
    logistic regression, in log-odds), relative to the average applicant."""
    model, info = MODELS[model_key], EXPLAIN[model_key]
    Z = model.named_steps["prep"].transform(complete(data))
    Z = Z.toarray() if hasattr(Z, "toarray") else np.asarray(Z)
    coef = model.named_steps["model"].coef_[0]
    phi = pd.DataFrame(coef * (Z - info["background_mean"]), columns=info["columns"], index=data.index)
    return phi.T.groupby(info["feature_of_column"], sort=False).sum().T


def strength(v):
    a = abs(v)
    return "strong" if a >= 0.5 else "moderate" if a >= 0.2 else "small"


# --------------------------------------------------------------------------------------
# UI building blocks
# --------------------------------------------------------------------------------------
def header():
    st.markdown(f'<div class="cp-header"><img src="{logo_data_uri()}" width="40" alt="">'
                '<span class="cp-title">CreditParity</span>'
                '<span class="cp-tagline">Profitable, Fair Lending Decisions</span></div>',
                unsafe_allow_html=True)


def page_intro(title, text):
    """Centered page title with a short explanation (same on every page)."""
    st.markdown(f'<div class="cp-intro"><div class="cp-page-title">{title}</div><p>{text}</p></div>',
                unsafe_allow_html=True)


def section_title(text, center=False):
    st.markdown(f'<div class="cp-section{" cp-center" if center else ""}">{text}</div>', unsafe_allow_html=True)


def card_grid(cards):
    """A row of equally high cards."""
    html = "".join(f'<div class="cp-card"><div class="cp-card-title">{t}</div><p>{x}</p></div>' for t, x in cards)
    st.markdown(f'<div class="cp-grid cp-grid-{len(cards)}">{html}</div>', unsafe_allow_html=True)


def zone_badge(zone, rating=None):
    z = ZONES[zone]
    rating_part = f"Risk rating {rating} · " if rating else ""
    return (f'<span class="cp-badge"><span class="cp-dot" style="background:{z["color"]}"></span>'
            f'{z["icon"]} {rating_part}{z["label"]}</span>')


def reasons_chart(contrib, row, top=6):
    order = contrib.abs().sort_values(ascending=False).index[:top]
    df = pd.DataFrame([{"Factor": f"{'↑' if contrib[f] > 0 else '↓'} {field_label(f)}: {value_label(f, row[f])}",
                        "Influence": float(contrib[f]),
                        "Effect": "Increases the risk" if contrib[f] > 0 else "Lowers the risk",
                        "Strength": strength(contrib[f])} for f in order])
    bars = alt.Chart(df).mark_bar(cornerRadiusEnd=4, height=20).encode(
        x=alt.X("Influence:Q", title="Influence on the risk", axis=alt.Axis(labels=False, ticks=False,
                gridColor="#ececea", titleColor=TEXT_SECONDARY)),
        y=alt.Y("Factor:N", sort=None, title=None, axis=alt.Axis(labelLimit=340, labelColor="#0b0b0b")),
        color=alt.Color("Effect:N", scale=alt.Scale(domain=["Increases the risk", "Lowers the risk"],
                                                    range=[RAISES_RISK, LOWERS_RISK]), legend=None),
        tooltip=["Factor", "Effect", "Strength"],
    )
    zero = alt.Chart(pd.DataFrame({"x": [0]})).mark_rule(color="#8c8c8c").encode(x="x:Q")
    return (bars + zero).properties(height=38 * len(df)).configure_view(stroke=None)


def analyst_summary(row, p, rating, zone, contrib):
    up = contrib[contrib > 0].sort_values(ascending=False).head(3)
    down = contrib[contrib < 0].sort_values().head(3)
    line = lambda f, v: f"  • {field_label(f)}: {value_label(f, row[f])} ({strength(v)})"
    return "\n".join([
        "APPLICATION SUMMARY – CreditParity",
        "",
        f"Loan: {row['LoanAmount']:,} for {row['LoanDuration']} months",
        f"Purpose: {value_label('LoanPurpose', row['LoanPurpose'])}",
        f"Chance of default: {p:.0%} · Risk rating {rating}",
        f"Recommendation: {ZONES[zone]['label']}",
        "",
        "Increases the risk:",
        *([line(f, v) for f, v in up.items()] or ["  • nothing significant"]),
        "",
        "Lowers the risk:",
        *([line(f, v) for f, v in down.items()] or ["  • nothing significant"]),
        "",
        "Please check:",
        "  • income, employment and account details",
        "  • whether the main risk factors are temporary",
        "  • age is only a fairness correction, never",
        "    a reason on its own",
        "",
        "Decision: [ ] approve  [ ] with conditions",
        "          [ ] decline",
        "Analyst: ____________",
    ])


# --------------------------------------------------------------------------------------
# Home
# --------------------------------------------------------------------------------------
def section_home():
    st.markdown(f"""
<div class="cp-hero">
  <img src="{logo_data_uri()}" width="96" alt="CreditParity logo">
  <div class="cp-hero-title">CreditParity</div>
  <p class="cp-tagline">Profitable, Fair Lending Decisions</p>
  <p class="cp-mission">Our mission is to help banks lend responsibly and fairly. CreditParity estimates how
  likely an applicant is to repay, explains the result in plain words and leaves uncertain cases to people.
  It does not use gender or nationality and corrects for hidden age bias, so that good customers are
  treated more fairly.</p>
</div>
""", unsafe_allow_html=True)

    section_title("What you can do", center=True)
    card_grid([
        ("📝 Check a single application", "Enter the details of one loan application and get the chance of "
                                          "default, a risk rating, a recommendation and the main reasons."),
        ("📂 Check several applications", "Upload a file with many applications and download the results "
                                          "for all of them at once."),
        ("⚙️ Bank settings", "Decide when applications are approved or declined automatically and see the effect "
                            "on workload, cost of wrong decisions and fairness."),
        ("💡 How it works", "Learn how CreditParity reaches its recommendations and what to keep in mind."),
    ])

    st.markdown('<div class="cp-author">Created by <b>Carina Domat</b></div>', unsafe_allow_html=True)


# --------------------------------------------------------------------------------------
# Check a single application
# --------------------------------------------------------------------------------------
def field_input(f):
    key = f"f_{f}"
    label = FIELD_LABELS[f]
    if f in CATEGORICAL:
        st.selectbox(label, list(VALUE_LABELS[f]), key=key, format_func=lambda v, f=f: VALUE_LABELS[f][v])
    elif f == "LoanAmount":
        lo, hi = META["numeric_ranges"][f]
        st.number_input(label, min_value=lo, max_value=hi, step=100, key=key)
    else:
        lo, hi = META["numeric_ranges"][f]
        st.slider(label, lo, hi, key=key)


def section_application():
    page_intro("Check a single application",
               "Enter the details of the loan application. The result on the right updates immediately.")

    left, right = st.columns([1, 1.15], gap="large")
    with left:
        for title, feats in FORM_GROUPS.items():
            with st.container(border=True):
                st.markdown(f'<div class="cp-card-title">{title}</div>', unsafe_allow_html=True)
                cols = st.columns(2)
                for i, f in enumerate(feats):
                    with cols[i % 2]:
                        field_input(f)
        st.caption("Gender and nationality are not asked for: CreditParity does not use them.")

    row = pd.DataFrame([{f: st.session_state[f"f_{f}"] for f in INPUT_FEATURES if f not in NOT_USED}])
    p = float(predict(row)[0])
    lower, upper, _ = current_policy()
    zone = zone_of(p, lower, upper).item()
    rating = rating_of(p)
    contrib = explain(row).iloc[0]

    with right:
        with st.container(border=True):
            st.markdown('<span class="cp-muted">Chance of default</span>', unsafe_allow_html=True)
            st.markdown(f'<div class="cp-pd">{p:.0%}</div>', unsafe_allow_html=True)
            st.markdown(zone_badge(zone, rating), unsafe_allow_html=True)
            st.markdown(f'<p style="margin-top:.7rem">{ZONES[zone]["advice"]}</p>', unsafe_allow_html=True)
            st.caption(f"Risk rating {rating} = {RATING_TEXT[rating]} (scale from A, lowest risk, to E, highest risk).")

        section_title("Main reasons for this result")
        st.altair_chart(reasons_chart(contrib, row.iloc[0]), width="stretch")
        st.caption("↑ Red bars increase the risk, ↓ blue bars lower it. Longer bars have more influence. "
                   "Age is only used to correct for age effects hidden in other information, so that "
                   "applicants are not treated unfairly because of their age.")

        if zone == "Yellow":
            section_title("Summary for the credit analyst")
            summary = analyst_summary(row.iloc[0], p, rating, zone, contrib)
            st.code(summary, language=None, wrap_lines=True)
            st.download_button("Download summary", summary, file_name="application_summary.txt")


# --------------------------------------------------------------------------------------
# Check several applications
# --------------------------------------------------------------------------------------
REQUIRED_COLUMNS = [f for f in INPUT_FEATURES if f not in NOT_USED]


@st.cache_data
def sample_csv():
    # 25 past applications spread over the whole range of risk
    test = evaluation[evaluation["split"] == "test"].sort_values("pd_final")
    ids = test.index[np.linspace(0, len(test) - 1, 25).astype(int)]
    return dataset.loc[ids, REQUIRED_COLUMNS].to_csv(index=False).encode()


def section_batch():
    page_intro("Check several applications",
               "Upload a CSV file with one application per row. "
               "Download the example file to see the expected format.")
    c1, c2 = st.columns([2, 1])
    with c1:
        uploaded = st.file_uploader("CSV file", type="csv", label_visibility="collapsed")
    with c2:
        st.download_button("Download example file (25 applications)", sample_csv(),
                           file_name="example_applications.csv", width="stretch")
        use_sample = st.toggle("Try it with the example file", value=uploaded is None, key="use_sample")

    if uploaded is not None:
        data = pd.read_csv(uploaded)
    elif use_sample:
        data = pd.read_csv(pd.io.common.BytesIO(sample_csv()))
    else:
        st.info("Upload a CSV file or try the example file.")
        return

    missing = [f for f in REQUIRED_COLUMNS if f not in data.columns]
    if missing:
        st.error("The file is missing these columns: " + ", ".join(missing) +
                 ". Please use the example file as a template.")
        return

    p = predict(data)
    lower, upper, _ = current_policy()
    zones = zone_of(p, lower, upper)
    contrib = explain(data)
    main_reason = [field_label(c.idxmax()) if c.max() > 0 else "–" for _, c in contrib.iterrows()]
    results = pd.DataFrame({
        "Application": np.arange(1, len(data) + 1),
        "Chance of default (%)": (100 * p).round(1),
        "Risk rating": [rating_of(v) for v in p],
        "Recommendation": [f"{ZONES[z]['icon']} {ZONES[z]['label']}" for z in zones],
        "Main reason for risk": main_reason,
        "Loan amount": data["LoanAmount"].to_numpy(),
        "Duration (months)": data["LoanDuration"].to_numpy(),
    })

    cols = st.columns(4)
    cols[0].metric("Applications", f"{len(results):,}", border=True)
    for col, z in zip(cols[1:], ZONE_ORDER):
        col.metric(f"{ZONES[z]['icon']} {ZONES[z]['label']}", f"{(zones == z).sum():,} ({(zones == z).mean():.0%})",
                   border=True)

    st.dataframe(results, hide_index=True, width="stretch", height=420,
                 column_config={"Chance of default (%)": st.column_config.ProgressColumn(
                     "Chance of default", format="%.0f%%", min_value=0, max_value=100)})
    download = pd.concat([results.drop(columns=["Application"]), data.reset_index(drop=True)], axis=1)
    st.download_button("Download results", download.to_csv(index=False).encode(),
                       file_name="creditparity_results.csv", type="primary")


# --------------------------------------------------------------------------------------
# Bank settings
# --------------------------------------------------------------------------------------
def policy_results(model_key, lower, upper):
    test = evaluation[evaluation["split"] == "test"]
    p = test["pd_final" if model_key == "d" else "pd_comparison"].to_numpy()
    y = test["y"].to_numpy()
    zone = zone_of(p, lower, upper)
    decline = p >= lower
    good51 = (test["age_group"].to_numpy() == "51+") & (y == 0)
    return {
        "Green": (zone == "Green").mean(), "Yellow": (zone == "Yellow").mean(), "Red": (zone == "Red").mean(),
        "Cost": int(5 * ((~decline) & (y == 1)).sum() + (decline & (y == 0)).sum()),
        "Caught": (decline & (y == 1)).sum() / (y == 1).sum(),
        "Good declined": int(((zone == "Red") & (y == 0)).sum()),
        "Bad approved": int(((zone == "Green") & (y == 1)).sum()),
        "Good 51+ declined": f"{int((good51 & (zone == 'Red')).sum())} of {int(good51.sum())}",
        "n_good": int((y == 0).sum()), "n_bad": int((y == 1).sum()), "n": len(y),
    }


GROUP_NAMES = {"female": "Women", "male": "Men", "19-25": "Age 19–25", "26-35": "Age 26–35",
               "36-50": "Age 36–50", "51+": "Age 51+", "no": "Not a foreign worker", "yes": "Foreign worker"}


def group_table(model_key, lower, upper):
    test = evaluation[evaluation["split"] == "test"].copy()
    p = test["pd_final" if model_key == "d" else "pd_comparison"].to_numpy()
    test["zone"] = zone_of(p, lower, upper)
    test["not_approved"] = p >= lower
    rows = []
    for col in ["Sex", "age_group", "ForeignWorker"]:
        for g, d in test.groupby(col):
            good = d[d["y"] == 0]
            rows.append({"Group": GROUP_NAMES.get(g, g), "Applications": len(d),
                         **{ZONES[z]["short"]: (d["zone"] == z).mean() for z in ZONE_ORDER},
                         "Good customers not approved automatically": good["not_approved"].mean()})
    return pd.DataFrame(rows)


def zone_share_chart(table):
    rows = []
    for _, r in table.iterrows():
        start = 0.0
        for z in ZONE_ORDER:
            share = r[ZONES[z]["short"]]
            rows.append({"Group": r["Group"], "Result": f"{ZONES[z]['icon']} {ZONES[z]['short']}",
                         "start": start, "end": start + share, "Share": f"{share:.0%}"})
            start += share
    df = pd.DataFrame(rows)
    order = list(dict.fromkeys(df["Group"]))
    labels = [f"{ZONES[z]['icon']} {ZONES[z]['short']}" for z in ZONE_ORDER]
    return alt.Chart(df).mark_bar(stroke=SURFACE, strokeWidth=2).encode(
        x=alt.X("start:Q", title=None, scale=alt.Scale(domain=[0, 1]),
                axis=alt.Axis(format="%", values=[0, .25, .5, .75, 1], gridColor="#ececea",
                              labelColor=TEXT_SECONDARY)),
        x2="end:Q",
        y=alt.Y("Group:N", sort=order, title=None, scale=alt.Scale(paddingInner=0.25),
                axis=alt.Axis(labelColor="#0b0b0b", labelLimit=200)),
        color=alt.Color("Result:N", sort=labels, scale=alt.Scale(domain=labels,
                        range=[ZONES[z]["color"] for z in ZONE_ORDER]),
                        legend=alt.Legend(orient="top", direction="horizontal", title=None, labelLimit=260,
                                         columnPadding=20)),
        tooltip=["Group", "Result", "Share"],
    ).properties(height=44 * len(order)).configure_view(stroke=None)


def section_settings():
    page_intro("Bank settings",
               "Decide when applications are handled automatically. The results show what these settings "
               "would have meant for 1,000 past applications.")

    rec_lower = to_percent(POLICY["models"]["d"]["lower_boundary"])
    rec_target = to_percent(POLICY["target_precision"])
    for state_key, widget_key in SETTING_WIDGETS.items():          # restore after switching pages
        if widget_key not in st.session_state:
            st.session_state[widget_key] = to_percent(st.session_state[state_key])

    controls, results = st.columns([1, 2], gap="large")
    with controls:
        with st.container(border=True):
            st.slider("Approve automatically when the chance of default is below", 5, 40, step=1,
                      key="w_lower_d", format="%d%%", on_change=persist, args=("lower_d",),
                      help=f"Recommended: {rec_lower}%. This value keeps the cost of wrong decisions lowest.")
            st.slider("Decline automatically only if at least this share of declines are truly bad loans",
                      70, 95, step=1, key="w_target", format="%d%%", on_change=persist, args=("target",),
                      help=f"Recommended: {rec_target}%. Higher values mean fewer good customers are declined "
                           "automatically, but more applications need a manual review.")
            lower, upper, _ = current_policy()
            st.markdown(f"→ Approved automatically below **{lower:.0%}** chance of default, declined "
                        f"automatically from **{upper:.0%}**, everything in between is reviewed by an analyst.")
            st.button("Back to recommended settings", on_click=reset_settings)

    r = policy_results("d", lower, upper)
    baseline_cost = policy_results("a", 0.5, 0.5)["Cost"]
    with results:
        c = st.columns(3)
        for col, z in zip(c, ZONE_ORDER):
            col.metric(f"{ZONES[z]['icon']} {ZONES[z]['short']}", f"{r[z]:.0%}", border=True)
        c = st.columns(2)
        c[0].metric("Cost of wrong decisions", f"{1 - r['Cost'] / baseline_cost:.0%} lower", border=True,
                    help="Compared with a common baseline: a credit model without fairness correction that "
                         "approves every application with a chance of default below 50%. A missed bad loan is counted as five times as costly as a turned-away good "
                         "customer. This is a weighted count of wrong decisions, not the bank's actual credit losses.")
        c[1].metric("Bad loans caught", f"{r['Caught']:.0%}", border=True,
                    help="Share of bad loans that are not approved automatically.")
        c = st.columns(2)
        c[0].metric("Good customers declined automatically", f"{r['Good declined']}", border=True,
                    help=f"Out of {r['n_good']} good customers among the {r['n']:,} past applications.")
        c[1].metric("Bad loans approved automatically", f"{r['Bad approved']}", border=True,
                    help=f"Out of {r['n_bad']} bad loans among the {r['n']:,} past applications.")

    section_title("Fairness check")
    st.markdown('<span class="cp-muted">How are different groups of applicants handled? In a fair process, '
                'similar shares of good customers should be approved in every group.</span>',
                unsafe_allow_html=True)
    table = group_table("d", lower, upper)
    st.altair_chart(zone_share_chart(table), width="stretch")
    left, right = st.columns([1, 1.6], gap="large")
    with left:
        st.metric("Good customers aged 51+ declined automatically", r["Good 51+ declined"], border=True)
    with right:
        show = table.set_index("Group")[["Good customers not approved automatically"]]
        st.dataframe(show.style.format("{:.0%}"), width="stretch")
        st.caption("Share of good customers in each group who are not approved automatically (they are reviewed "
                   "or declined). Groups with few applicants (not a foreign worker, age 51+) are less reliable.")


# --------------------------------------------------------------------------------------
# How it works
# --------------------------------------------------------------------------------------
def section_about():
    page_intro("How it works",
               "How CreditParity reaches its recommendations, how to read them and what to keep in mind.")

    section_title("From application to recommendation")
    card_grid([
        ("1 · Estimate the risk", "CreditParity compares the application with thousands of past loans and "
                                  "estimates the chance that the loan will not be repaid."),
        ("2 · Sort into three groups", "Low-risk applications are approved automatically, high-risk "
                                       "applications are declined, and uncertain ones go to a credit analyst."),
        ("3 · People decide", "Analysts make the final decision on uncertain cases, and every declined "
                              "applicant can ask for a personal review."),
    ])

    lower, upper, _ = current_policy()
    section_title("Risk rating and recommendation")
    st.markdown("The risk rating (A–E) describes the applicant's risk and depends only on the chance of default; "
                "the recommendation (approve, review, decline) describes how the bank handles the application and "
                "follows the bank settings, so changing the settings can change the recommendation but never the "
                "rating.")
    c1, c2 = st.columns(2)
    c1.dataframe(pd.DataFrame([{"Risk rating": r,
                                "Chance of default": f"{lo:.0%} – {hi:.0%}" if hi <= 1 else f"{lo:.0%} or more",
                                "Meaning": RATING_TEXT[r]} for r, lo, hi in RATING_SCALE]),
                 hide_index=True, width="stretch")
    c2.dataframe(pd.DataFrame([
        {"Recommendation": f"{ZONES['Green']['icon']} {ZONES['Green']['label']}",
         "Chance of default": f"below {lower:.0%}"},
        {"Recommendation": f"{ZONES['Yellow']['icon']} {ZONES['Yellow']['label']}",
         "Chance of default": f"{lower:.0%} – {upper:.0%}"},
        {"Recommendation": f"{ZONES['Red']['icon']} {ZONES['Red']['label']}",
         "Chance of default": f"{upper:.0%} or more"},
    ]), hide_index=True, width="stretch")

    section_title("Our fairness promise")
    st.markdown(
        "- **Gender and nationality are never used.** Two applicants who differ only in these points get the same "
        "result.\n"
        "- **Hidden age bias is corrected.** Age is used only to offset age effects hidden in other information. "
        "With the recommended settings, none of the 21 creditworthy applicants aged 51+ among the past applications "
        "was declined automatically; uncertain cases go to a person instead.\n"
        "- **Every result is explained.** Analysts see the main reasons behind each recommendation.")

    section_title("Good to know")
    st.markdown(
        "- The results are **estimates, not certainties**. They are based on example data, not on real customers.\n"
        "- To correct for age, CreditParity rates **young applicants as riskier** than their group's actual track "
        "record (about 4 times) and **older applicants as less risky**. Most young applicants affected by this are "
        "sent to an analyst rather than declined (only about 2.5% are declined automatically).\n"
        "- About **6 in 10 applications go to a credit analyst** with the recommended settings. The bank settings "
        "can reduce this workload, at the price of more automatic declines.\n"
        "- Even without using gender as an input, **men are approved automatically less often than women**, because "
        "their other information differs on average.\n"
        "- The recommended settings are business choices and should be **confirmed by the bank's risk and "
        "compliance teams** before real use. A legal review is needed before CreditParity is used with real "
        "customers.\n"
        "- The explanations are meant for **bank staff**. A version written for applicants is planned.")


# --------------------------------------------------------------------------------------
# Navigation
# --------------------------------------------------------------------------------------
SECTIONS = {"home": "🏠 Home", "single": "📝 Check a single application", "batch": "📂 Check several applications",
            "bank": "⚙️ Bank settings", "info": "💡 How it works"}


def main():
    init_state()
    st.logo(str(LOGO_PATH), size="large")
    requested = st.query_params.get("section", "home")
    keys = list(SECTIONS)
    with st.sidebar:
        st.markdown("### CreditParity")
        choice = st.radio("Section", keys, index=keys.index(requested) if requested in keys else 0,
                          format_func=SECTIONS.get, label_visibility="collapsed")
        st.query_params["section"] = choice
        if choice != "home":
            st.divider()
            lower, upper, _ = current_policy()
            st.markdown("**Current settings**")
            st.markdown(f"{ZONES['Green']['icon']} Approve below {lower:.0%}  \n"
                        f"{ZONES['Yellow']['icon']} Review {lower:.0%} – {upper:.0%}  \n"
                        f"{ZONES['Red']['icon']} Decline from {upper:.0%}")
            st.caption("Change them under Bank settings.")

    if choice != "home":
        header()
    {"home": section_home, "single": section_application, "batch": section_batch,
     "bank": section_settings, "info": section_about}[choice]()


main()
