# AI Usage

This project was built with help from **Claude Code** (Anthropic, model Claude Opus 5.5), an AI coding assistant. This file documents what the assistant was used for and how we verified its output. The team designed the project idea, business case, cost assumption and three-zone policy, and reviewed every step before it was accepted.

## How we worked

- The assistant worked step by step. After each step it summarized the results, and the team reviewed and approved them before the next step started.
- All code was executed and every number reported in the notebooks and README comes from an actual run. No result was typed in by hand without being produced by code.
- The Steps 1–3 notebook already existed when the assistant was brought in.

## Log

| Step | What the assistant did | How it was verified |
|---|---|---|
| Repo setup | Created the folder structure, `.gitignore`, `requirements.txt`, a local Python environment and this file. Re-ran the existing Steps 1–3 notebook in a scratch copy. | The re-run reproduced all reported Steps 1–3 numbers (e.g. accuracy 0.800, recall 0.626, ROC AUC 0.867; CV ROC AUC 0.859 vs. 0.848). |
| Step 4 – Model comparison | Wrote and ran `notebooks/credit_risk_step_4_model_comparison.ipynb`: logistic regression, random forest and XGBoost with 5-fold stratified CV on the training set, default and tuned (nested CV); drafted the interpretation. | Notebook executed end to end; the numbers in the text were checked against the printed tables. Per-fold comparison confirms that logistic regression is best in 5 of 5 folds. Team reviewed and approved the model decision. |
| Step 5 – Thresholds and zones | Wrote and ran `notebooks/credit_risk_step_5_thresholds_and_zones.ipynb`: out-of-fold predictions, cost curve and cost-optimal threshold, three-zone policy (shares, defaults and good customers per zone), analyst workload for different upper boundaries, per-fold checks, final test evaluation. After team review, the notebook was simplified: the team decided to justify the zones by human oversight and workload rather than by a cost model for manual review. | Notebook executed end to end; every number in the text was checked against the printed tables. Test results confirmed the 0.5 baseline from Step 3 (TP 209, TN 591, FP 75, FN 125). Team reviewed and approved the threshold (0.11) and zone boundaries (0.11 / 0.37). |
| Step 6 – Fairness | Wrote and ran `notebooks/credit_risk_step_6_fairness.ipynb`: group metrics (rejection rate, recall, FPR, DPD, EOD) with fairlearn for three variants (all features, without sensitive features, reweighing), proxy analysis, comparison with fairlearn's CorrelationRemover, zone shares per group, test-set confirmation. Explored ExponentiatedGradient, which was not usable with the cost matrix, and reported this. | Notebook executed end to end; every number in the text was checked against the printed tables. Out-of-fold results and test-set results were compared for consistency. Team review of the fairness results and the choice of final model: pending. |
