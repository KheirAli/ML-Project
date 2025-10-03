
# Problem Definition:

**Problem**
Detect **fraudulent transactions** in AMLSim logs while revealing *how* they occur via **8 laundering motifs** (fan-out, fan-in, scatter-gather, gather-scatter, simple cycle, random walk, bipartite, stack). Each transaction should receive (i) a **fraud score** and (ii) an **8-dimensional motif-proximity signature** for interpretability.

**Motivation**
Traditional anomaly detectors focus on isolated edges and miss **structural, time-ordered patterns** of laundering. Compliance teams need **interpretable** alerts (“near fan-in + rapid aggregation”) and **low false-positive rates**. Fields like *credit-card type* or *reinvestment flags* are often weak signals; the **graph+temporal structure** carries the core signal. Our approach integrates **structure-aware, time-windowed features** with **motif proximity** to boost accuracy and explainability.

---

# Methods:

**Data Preprocessing Methods Identified**

1. **Temporal windowing & leakage control:** build past-only sliding windows (e.g., 7–30 days) to compute features without future leakage.
2. **Graph construction:** directed multigraph of accounts (nodes) and transactions (edges); 1–2-hop ego-nets per edge.
3. **Feature engineering:**

   * Topology/role stats (in/out-degree, unique counterparties, entropy), temporal gaps/burstiness, short cycles, fan ratios, path-continuation likelihood.
   * **Aux categorical encoding** (credit-card/reinvestment) via NLP embeddinga.
4. **Scaling & regularization:** Standardize continuous features (e.g., `sklearn.preprocessing.StandardScaler`), shrinkage on covariance for distance models.
5. **Class imbalance handling:** weighted loss or focal loss; stratified time splits.

**ML Algorithms/Models Identified**

* **Unsupervised / Representation**

  * **Prototype distances (Mahalanobis w/ shrinkage):** per-motif ((\mu_k,\Sigma_k)) → softmax over distances. (`numpy/scipy` for covariance; custom light wrapper)
  * **K-Means / GMM** for constrained clustering of motif prototypes. (`sklearn.cluster.KMeans`, `sklearn.mixture.GaussianMixture`)
* **Supervised (Tabular)**

  * **Logistic Regression** with calibration. (`sklearn.linear_model.LogisticRegression` + `sklearn.isotonic.IsotonicRegression` or `sklearn.calibration.CalibratedClassifierCV`)
  * **Gradient-Boosted Trees** (XGBoost/LightGBM). (`xgboost.XGBClassifier` / `lightgbm.LGBMClassifier`)
* **Graph-Native (Supervised)**

  * **GNN baselines** for edge classification with temporal context: GraphSAGE/GIN; **temporal GNNs** (TGAT/TGN). (`pyg`/PyTorch Geometric: `torch_geometric.nn.SAGEConv`, `GINConv`; TGAT/TGN implementations)

**Why this is effective**

* **Structure-aware:** Graph+temporal features capture the real laundering mechanism.
* **Interpretable:** 8-way motif proximity explains *why* a transaction is suspicious.
* **Performant:** Supervised models learn interactions between motif proximity and raw features; GNNs exploit neighborhood context for SOTA accuracy.
* **Robust:** Past-only windows and calibration reduce leakage and overconfidence.

---


**Quantitative Metrics**

* **Detection:** AUROC, **Average Precision (PR-AUC)**, **Recall@FPR=1%** (ops-friendly), F1, **Brier score** + **Expected Calibration Error (ECE)**.
* **Clustering/Representation:** **ARI**, **NMI**, Silhouette (for motif separation).
* **Compute/Latency:** per-window feature time, per-edge scoring time.

**Project Goals**

* **Accuracy & Reliability:**

  * +3–7 **AUROC** points vs. raw-features baseline;
  * **+5–10% Recall@FPR=1%**;
  * **ECE ≤ 0.05** after calibration for trustworthy probabilities.
* **Interpretability:** Every alert includes motif rationale (e.g., “fan-in proximity 0.81 + rapid aggregation”).
* **Ethics & governance:**

  * Minimize investigator burden (**lower FPR**) to reduce unnecessary account holds.
  * Auditability: store motif signatures and SHAP attributions for decisions.
  * **Bias checks:** monitor subgroup FPR/TPR across channels/regions; drop spurious categorical signals if they induce drift or bias.


**Expected Results**

* **Motif separation:** ARI/NMI ≫ random; clear UMAP clusters in distance/signature space.
* **Detection lift:** Adding motif signature (`p[8]`, hard label `c`) to raw features improves **PR-AUC** notably in the low-FPR regime.
* **Calibration:** Post-training isotonic/Platt calibration yields **well-calibrated** risk scores (tight reliability curves).
* **Operational readiness:** Linear-time feature extraction, cached prototypes, and sub-millisecond per-edge scoring for tabular models; GNN used selectively where graph context is essential.


Here’s a compact section you can drop into your report. It stays lean to help you fit under the 800-word cap.

---

## (5) Project Management & Contributions

### Gantt Chart (Fall → Spring, responsibilities by member)

**Fall (Weeks 1–14)**

* **W1–2:** Data ingestion, schema audit, leakage checks — **Alireza** (lead), Member2 (support)
* **W3–4:** Temporal windowing, graph build, EDA — **Member2** (lead), Alireza (review)
* **W5–6:** Feature set v1 (role/temporal/motif), scaling/encoders — **Member3** (lead)
* **W7–8:** Prototype fitting (μ, Σ, shrinkage), soft motif scores — **Alireza** (lead)
* **W9–10:** Tabular baselines (LR/GBDT), imbalance & calibration — **Member2** (lead)
* **W11–12:** GNN baseline (TGAT/TGN or SAGE), past-only neighborhoods — **Member3** (lead)
* **W13:** Ablations, SHAP/explanations, reliability curves — **Alireza** (lead)
* **W14:** Interim report + slides — **All** (Alireza owner)

**Spring (Weeks 1–10)**

* **W1–2:** Error analysis, feature v2, prototype retune — **Member2**
* **W3–4:** Efficiency pass (profiling, caching, batching) — **Member3**
* **W5–6:** Robustness & bias checks (subgroup FPR/TPR) — **Alireza**
* **W7–8:** Documentation, repo polish, figs — **All** (Member2 owner)
* **W9–10:** Final write-up & demo — **All** (Member3 owner)

> Deliverables per milestone: PR with code + brief README; tracked in Issues/Projects. Owners above ensure reviews and weekly status notes.

### Contribution Table (Proposal stage)

| Name                    | Proposal Contributions                                                                                                                         |
| ----------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| **Alireza Kheirandish** | Drafted problem/motivation; designed motif-proximity approach; wrote Methods outline; created evaluation plan and ethics/sustainability notes. |
| **Member2**             | Spec’d preprocessing pipeline (windows, encoders); selected tabular baselines and calibration plan; organized repo structure.                  |
| **Member3**             | Planned GNN baseline and leakage controls; defined ablations, visuals (UMAP/SHAP); edited for clarity and word count.                          |

> **Template (edit as needed):**
>
> * Replace names/roles above.
> * If a 2-person team, merge W11–12 duties into Member2 or Alireza.
> * Keep this section concise: bullets + table minimize word usage while satisfying requirements.

