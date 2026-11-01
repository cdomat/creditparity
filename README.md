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
| 🟢 Green – auto-approve (PD < 0.16) | Low default probability, approved automatically |
| 🟡 Yellow – manual review (0.16 ≤ PD < 0.65) | Uncertain cases: a credit analyst decides, supported by a SHAP explanation. This is the only manual workload. |
| 🔴 Red – decline recommended (PD ≥ 0.65) | The system declines and gives the customer a SHAP-based explanation; the customer can request a human review |

The zones are justified by **human oversight**, not by extra cost savings: the EU AI Act classifies credit scoring as a high-risk AI system that requires effective human oversight, and GDPR Art. 22 gives applicants the right to human intervention and to contest automated decisions. At the same time, the zones keep the analyst workload manageable by handling clear cases automatically.

**Boundaries.** The lower boundary is the cost-optimal threshold (5:1 cost matrix). The upper boundary is the lowest threshold at which at least 80% of automatic declines are actual defaults (precision rule). Both are chosen on cross-validated predictions of the training data.

**Final model.** A logistic regression trained with **reweighing** on age group, sex and foreign worker status, so that it does not learn the group differences in default rates. The same model without reweighing is kept as a comparison to show the price of fairness.

**Core message:** a simple, explainable model performs as well as a complex one. The real gain comes from catching more defaults with a cost-based threshold, combined with human judgment for uncertain cases. With the fair model, the cost on the test set is still about **30% lower** than with the default threshold of 0.5 (488 vs. 700), while the age gap for creditworthy applicants drops from 0.80 to 0.26 and the sex gap from 0.15 to 0.04.

### The price of fairness

Compared with model (a), the fair model (c):
- costs **19% more** (488 vs. 409 on the test set), but is still **30% cheaper** than the default threshold of 0.5 (700);
- sends more applications to manual review (**53% vs. 42%**);
- auto-approves more defaults (**25 vs. 14**).

In return:
- **no creditworthy applicant aged 51+ is declined automatically** on the test set (0 of 21, vs. 8 of 21 with model (a); on the training data 5 of 83 vs. 41 of 83);
- the zone shares of women and men are almost identical (out-of-fold: green 36% vs. 32%, yellow 51% vs. 52%, red 13% vs. 16%);
- the age gap for creditworthy applicants (EOD) falls from 0.80 to 0.26 and the sex gap from 0.15 to 0.04.

## Data

- `data/german_credit_data_biased_training.csv`: IBM version of the German Credit dataset, **synthetic**, 5,000 applications, 20 features, no missing values
- Target: `Risk` (default = 1, about 33% of applications)
- Sensitive attributes: `Sex`, `Age`, `ForeignWorker`

## Results so far

| Step | Notebook | Key result |
|---|---|---|
| 1–3 Business case, EDA, baseline | `notebooks/credit_risk_steps_1-3.ipynb` | Logistic regression on the test set: accuracy 0.80, recall 0.63, ROC AUC 0.87. Strong age effect in the data (default rate 7% for ages 19–25 vs. 78% for 51+). |
| 4 Model comparison | `notebooks/credit_risk_step_4_model_comparison.ipynb` | 5-fold CV on the training set: logistic regression ROC AUC **0.857**, tuned XGBoost 0.850, tuned random forest 0.848. Logistic regression is best in all 5 folds, so we keep the simple, explainable model. |
| 5 Thresholds and zones | `notebooks/credit_risk_step_5_thresholds_and_zones.ipynb` | Final model (c), thresholds chosen on out-of-fold predictions of the training set. Lower boundary = cost-optimal threshold **0.16**; upper boundary **0.65** (≥ 80% of automatic declines are defaults). Test set: cost **700 → 488 (−30%)** vs. the default threshold 0.5, recall 0.63 → 0.93; 33% auto-approved, 53% manual review, 14% decline recommended (red-zone precision 82.5%, 25 good customers declined automatically). Comparison model (a): cost 409. |
| 6 Fairness | `notebooks/credit_risk_step_6_fairness.ipynb` | Model (a) declines 95–96% of creditworthy applicants aged 51+ vs. 14–17% of those aged 19–25 (Equal Opportunity Difference 0.80 out-of-fold). **Dropping sensitive features does not help** (EOD 0.82), because age can be reconstructed from other features (ROC AUC 0.90). **Reweighing** (final model) cuts the age gap to 0.26 and the sex gap from 0.15 to 0.04, at 17–19% higher cost than model (a). Good customers aged 51+ declined automatically: 41 of 83 with (a), 5 of 83 with (c). |
| 7 Explainability | `notebooks/credit_risk_step_7_explainability.ipynb` | Exact SHAP values for the logistic regression, aggregated to the 20 input features. In the fair model, **age is the strongest driver, with a negative coefficient**: it acts as a counterweight to the age proxies (e.g. long employment). Age appears among the top 3 reasons in 65% of explanations, which matters for customer-facing explanations. Some explanations are implausible because the data is synthetic (e.g. a negative checking balance lowers the risk). |

An initial analysis with IBM watsonx AutoAI selected an XGBoost pipeline by accuracy (79.6% on a 499-case holdout, but recall of only 53%). Our fair comparison on identical folds shows that this extra complexity does not pay off on this dataset.

## Limitations

- The dataset is **synthetic**. Some relationships are implausible, e.g. a negative checking account balance (`CheckingStatus = less_0`) has the lowest default rate. Results may not transfer to real loan data.
- The **5:1 cost ratio** is an assumption, not an estimate from real loss data.
- The 80% precision target for automatic declines is a business assumption. Before deployment it should be validated with the bank's risk and compliance teams. In operation, the thresholds should be monitored regularly and adjusted to the available analyst capacity: a higher target reduces wrongful automatic declines but increases the manual review workload.
- With this target, about half of all applications (53% on the test set) go to manual review, a substantial workload.
- The fair model auto-approves more defaults than model (a) (25 vs. 14 on the test set), because its lower boundary is higher (0.16 vs. 0.11) and it separates defaults from good customers less sharply.
- Cost figures assume that every application with a default probability ≥ 0.16 is declined. Analyst decisions in the yellow zone and human reviews requested in the red zone may change some decisions; the dataset cannot show how this affects cost.
- SHAP explains the model, not the causes of default. In the fair model, age (and to a lesser extent sex) appear among the main drivers of individual explanations, so customer-facing explanations need careful wording and a legal review.
- Small groups: only 123 applicants have `ForeignWorker = no` (23 in the test set), so fairness metrics for this group are uncertain.
- Reweighing uses protected attributes (age, sex, foreign worker status) during training, which needs a legal review.
- Reweighing increases declines of young good customers: 41% of good customers aged 19–25 have a default probability above the lower boundary (out-of-fold; 45% on the test set), compared with 17% without reweighing. Most of them go to manual review; 3% are declined automatically.

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
