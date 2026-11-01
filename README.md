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
| 🟢 Green – auto-approve (PD < 0.15) | Low default probability, approved automatically |
| 🟡 Yellow – manual review (0.15 ≤ PD < 0.71) | Uncertain cases: a credit analyst decides, supported by a SHAP explanation. This is the only manual workload. |
| 🔴 Red – decline recommended (PD ≥ 0.71) | The system declines and gives the customer a SHAP-based explanation; the customer can request a human review |

The zones are justified by **human oversight**, not by extra cost savings: the EU AI Act classifies credit scoring as a high-risk AI system that requires effective human oversight, and GDPR Art. 22 gives applicants the right to human intervention and to contest automated decisions. At the same time, the zones keep the analyst workload manageable by handling clear cases automatically.

**Boundaries.** The lower boundary is the cost-optimal threshold (5:1 cost matrix). The upper boundary is the lowest threshold at which at least 80% of automatic declines are actual defaults (precision rule). Both are chosen on cross-validated predictions of the training data.

**Final model (d).** A logistic regression trained with **reweighing** on age group, sex and foreign worker status, so that it does not learn the group differences in default rates. **Sex and foreign worker status are not model inputs**: no applicant's score depends directly on sex. `Age` stays as an input, as a correction for the many features that indirectly reveal age. The model with all features and no reweighing, (a), is kept as a comparison to show the price of fairness.

**Core message:** a simple, explainable model performs as well as a complex one. The real gain comes from catching more defaults with a cost-based threshold, combined with human judgment for uncertain cases. With the fair model, the cost on the test set is still about **one third lower** than with the default threshold of 0.5 (467 vs. 700), while the age gap for creditworthy applicants drops from 0.81 to 0.37.

### The price of fairness

Compared with model (a), the fair model (d):
- costs **14% more** (467 vs. 409 on the test set), but is still **33% cheaper** than the default threshold of 0.5 (700);
- sends more applications to manual review (**60% vs. 42%**);
- auto-approves more defaults (**17 vs. 14**);
- has a larger sex gap among good customers (0.17 vs. 0.11 on the test set), which works against men.

In return:
- **no creditworthy applicant aged 51+ is declined automatically** on the test set (0 of 21, vs. 8 of 21 with model (a); on the training data 1 of 83 vs. 41 of 83);
- the age gap for creditworthy applicants (EOD) falls from 0.81 to 0.37 (test set);
- no applicant's score depends directly on sex or foreign worker status.

### Equal treatment vs. equal outcomes for sex

We also tested model (c), which uses sex as an input. It almost removes the sex gap (0.03 on the test set), but only because, with all other features equal, it rates women as riskier than men. Every model that scores men and women with the same features identically shows a sex gap of about 0.2 against men, because men in this dataset have less favorable values on other features. CreditParity chooses **equal treatment**: no direct sex-based treatment is a core principle.

## Data

- `data/german_credit_data_biased_training.csv`: IBM version of the German Credit dataset, **synthetic**, 5,000 applications, 20 features, no missing values
- Target: `Risk` (default = 1, about 33% of applications)
- Sensitive attributes: `Sex`, `Age`, `ForeignWorker`

## Results so far

| Step | Notebook | Key result |
|---|---|---|
| 1–3 Business case, EDA, baseline | `notebooks/credit_risk_steps_1-3.ipynb` | Logistic regression on the test set: accuracy 0.80, recall 0.63, ROC AUC 0.87. Strong age effect in the data (default rate 7% for ages 19–25 vs. 78% for 51+). |
| 4 Model comparison | `notebooks/credit_risk_step_4_model_comparison.ipynb` | 5-fold CV on the training set: logistic regression ROC AUC **0.857**, tuned XGBoost 0.850, tuned random forest 0.848. Logistic regression is best in all 5 folds, so we keep the simple, explainable model. |
| 5 Thresholds and zones | `notebooks/credit_risk_step_5_thresholds_and_zones.ipynb` | Final model (d), thresholds chosen on out-of-fold predictions of the training set. Lower boundary = cost-optimal threshold **0.15**; upper boundary **0.71** (≥ 80% of automatic declines are defaults). Test set: cost **700 → 467 (−33%)** vs. the default threshold 0.5, recall 0.63 → 0.95; 30% auto-approved, 60% manual review, 10% decline recommended (red-zone precision 86%, 14 good customers declined automatically). Comparison model (a): cost 409. |
| 6 Fairness | `notebooks/credit_risk_step_6_fairness.ipynb` | Model (a) declines 95–96% of creditworthy applicants aged 51+ vs. 14–17% of those aged 19–25 (Equal Opportunity Difference 0.80 out-of-fold). **Dropping sensitive features does not help** (EOD 0.82), because age can be reconstructed from other features (ROC AUC 0.90). **Reweighing with age as an input** (final model (d)) cuts the age gap to 0.29. Good customers aged 51+ declined automatically: 41 of 83 with (a), 1 of 83 with (d). For sex, equal treatment and equal outcomes conflict on this data (models (c) and (e) as evidence); CreditParity chooses equal treatment. |
| 7 Explainability | `notebooks/credit_risk_step_7_explainability.ipynb` | Exact SHAP values for the logistic regression, aggregated to the input features. In the fair model, **age is the strongest driver, with a negative coefficient**: it acts as a counterweight to the age proxies (e.g. long employment), so it is shown to analysts as "Age (fairness correction)". Sex and foreign worker status never appear. Some explanations are implausible because the data is synthetic (e.g. a negative checking balance lowers the risk). |
| 8 Saved model | `notebooks/credit_risk_step_8_save_model.ipynb` | `model/creditparity_bundle.joblib` holds the final model (d) and the comparison model (a) as complete sklearn pipelines, the decision policy (0.15 / 0.71, 80% precision target, 5:1 costs) and the data for exact SHAP explanations; `model/policy.json` and `model/evaluation_data.csv` support the dashboard. Reloading the files in a separate process reproduces all 26 checked test-set values. |

An initial analysis with IBM watsonx AutoAI selected an XGBoost pipeline by accuracy (79.6% on a 499-case holdout, but recall of only 53%). Our fair comparison on identical folds shows that this extra complexity does not pay off on this dataset.

## Limitations

- The dataset is **synthetic**. Some relationships are implausible, e.g. a negative checking account balance (`CheckingStatus = less_0`) has the lowest default rate. Results may not transfer to real loan data.
- The **5:1 cost ratio** is an assumption, not an estimate from real loss data.
- The 80% precision target for automatic declines is a business assumption. Before deployment it should be validated with the bank's risk and compliance teams. In operation, the thresholds should be monitored regularly and adjusted to the available analyst capacity: a higher target reduces wrongful automatic declines but increases the manual review workload.
- About 60% of applications go to manual review. The bank can adjust the precision target to balance analyst workload against wrongful automatic declines; thresholds should be monitored and adjusted to the available analyst capacity.
- The fair model auto-approves more defaults than model (a) (17 vs. 14 on the test set), because its lower boundary is higher (0.15 vs. 0.11) and it separates defaults from good customers less sharply.
- Cost figures assume that every application with a default probability ≥ 0.15 is declined. Analyst decisions in the yellow zone and human reviews requested in the red zone may change some decisions; the dataset cannot show how this affects cost.
- The model does not use sex or foreign worker status as inputs. The remaining sex gap of about 0.2 among good customers works against men and results from differences in other features. It should be reviewed legally (direct vs. indirect discrimination) and monitored in operation.
- Age is still used directly as a correction for its proxies. This also needs legal review.
- Reweighing uses age group, sex and foreign worker status to compute the training weights, which also belongs in the legal review.
- Reweighing increases declines of young good customers: 42% of good customers aged 19–25 have a default probability above the lower boundary (out-of-fold; 49% on the test set), compared with 17% without reweighing. Most of them go to manual review; 2.5% are declined automatically.
- Explanations are analyst-facing only. A customer-facing explanation for automatic declines (GDPR Art. 22 right to meaningful information) is a next step that requires legal review.
- SHAP explains the model, not the causes of default.
- Small groups: only 123 applicants have `ForeignWorker = no` (23 in the test set), so fairness metrics for this group are uncertain.

## Repository structure

```
data/        dataset
notebooks/   one notebook per project phase (Google Colab compatible)
model/       final and comparison pipelines, decision policy, evaluation data (Step 8)
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
