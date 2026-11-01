# CreditParity – Profitable, Fair Lending Decisions

CreditParity is a credit risk scoring prototype for a bank's retail customers that aims to be both **profitable** and **demonstrably fair**. The name refers to *Demographic Parity*, one of the fairness metrics we measure.

A user enters a single loan application (or uploads several as a CSV) and receives a default probability, a rating class, a decision zone and an explanation of the main drivers.

> **Live app:** _link will be added after deployment on Streamlit Community Cloud (Step 10)._

*Final team project, BUS/FIN 241A Machine Learning, Brandeis University.*

## Business case

For every loan application, a bank decides whether to approve it. Both possible errors cost money, but not equally:

- **Missed default (false negative):** a defaulter is approved, and the bank loses a large part of the loan.
- **Rejected good customer (false positive):** the bank only loses the interest margin.

We assume that a missed default costs **5 times** as much as a rejected good customer (the classic cost matrix for the German Credit data). This ratio is an assumption and can be changed.

Instead of a simple approve/reject decision, CreditParity uses a **human-in-the-loop policy with three zones**:

| Zone | Meaning |
|---|---|
| 🟢 Green – auto-approve (PD < 0.11) | Low default probability, approved automatically |
| 🟡 Yellow – manual review (0.11 ≤ PD < 0.37) | Uncertain cases: a credit analyst decides, supported by a SHAP explanation. This is the only manual workload. |
| 🔴 Red – decline recommended (PD ≥ 0.37) | The system declines and gives the customer a SHAP-based explanation; the customer can request a human review |

The zones are justified by **human oversight**, not by extra cost savings: the EU AI Act classifies credit scoring as a high-risk AI system that requires effective human oversight, and GDPR Art. 22 gives applicants the right to human intervention and to contest automated decisions. At the same time, the zones keep the analyst workload manageable by handling clear cases automatically.

**Core message:** a simple, explainable model performs as well as a complex one. The real gain comes from catching more defaults with a cost-based threshold, combined with human judgment for uncertain cases.

## Data

- `data/german_credit_data_biased_training.csv`: IBM version of the German Credit dataset, **synthetic**, 5,000 applications, 20 features, no missing values
- Target: `Risk` (default = 1, about 33% of applications)
- Sensitive attributes: `Sex`, `Age`, `ForeignWorker`

## Results so far

| Step | Notebook | Key result |
|---|---|---|
| 1–3 Business case, EDA, baseline | `notebooks/credit_risk_steps_1-3.ipynb` | Logistic regression on the test set: accuracy 0.80, recall 0.63, ROC AUC 0.87. Strong age effect in the data (default rate 7% for ages 19–25 vs. 78% for 51+). |
| 4 Model comparison | `notebooks/credit_risk_step_4_model_comparison.ipynb` | 5-fold CV on the training set: logistic regression ROC AUC **0.857**, tuned XGBoost 0.850, tuned random forest 0.848. Logistic regression is best in all 5 folds, so we keep the simple, explainable model. |
| 5 Thresholds and zones | `notebooks/credit_risk_step_5_thresholds_and_zones.ipynb` | **Headline:** the cost-based threshold of **0.11** (chosen on out-of-fold predictions of the training set) lowers the test-set cost by **42% (700 → 409)** and raises recall from **0.63 to 0.96**. Zones on the test set: 34% auto-approved, 27% manual review (analyst workload), 39% decline recommended. |

An initial analysis with IBM watsonx AutoAI selected an XGBoost pipeline by accuracy (79.6% on a 499-case holdout, but recall of only 53%). Our fair comparison on identical folds shows that this extra complexity does not pay off on this dataset.

## Limitations

- The dataset is **synthetic**. Some relationships are implausible, e.g. a negative checking account balance (`CheckingStatus = less_0`) has the lowest default rate. Results may not transfer to real loan data.
- The **5:1 cost ratio** is an assumption, not an estimate from real loss data.
- Cost figures assume that every application with a default probability ≥ 0.11 is declined. Analyst decisions in the yellow zone and human reviews requested in the red zone may change some decisions; the dataset cannot show how this affects cost.
- The upper zone boundary (0.37) is a workload choice, not an optimum.
- Small groups: only 123 applicants have `ForeignWorker = no`, so fairness metrics for this group are uncertain.

## Repository structure

```
data/        dataset
notebooks/   one notebook per project phase (Google Colab compatible)
model/       saved final pipeline and zone thresholds (Step 8)
app.py       Streamlit dashboard (Step 9)
AI_USAGE.md  documentation of AI assistance
```

## Running the notebooks locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt xgboost jupyter
jupyter notebook notebooks/
```

In Google Colab, upload the CSV and set `DATA_PATH = "german_credit_data_biased_training.csv"` in the loading cell. XGBoost is pre-installed in Colab. On macOS, XGBoost needs the OpenMP runtime (`brew install libomp`).
