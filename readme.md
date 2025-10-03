
# Problem Definition:

✅ **Problem**
Detect **fraudulent transactions** in AMLSim logs while revealing *how* they occur via **8 laundering motifs** (fan-out, fan-in, scatter-gather, gather-scatter, simple cycle, random walk, bipartite, stack). Each transaction should receive (i) a **fraud score** and (ii) an **8-dimensional motif-proximity signature** for interpretability.

✅ **Motivation**
Traditional anomaly detectors focus on isolated edges and miss **structural, time-ordered patterns** of laundering. Compliance teams need **interpretable** alerts (“near fan-in + rapid aggregation”) and **low false-positive rates**. Fields like *credit-card type* or *reinvestment flags* are often weak signals; the **graph+temporal structure** carries the core signal. Our approach integrates **structure-aware, time-windowed features** with **motif proximity** to boost accuracy and explainability.

---

# Methods:

✅ **3+ Data Preprocessing Methods Identified**

1. **Temporal windowing & leakage control:** build past-only sliding windows (e.g., 7–30 days) to compute features without future leakage.
2. **Graph construction:** directed multigraph of accounts (nodes) and transactions (edges); 1–2-hop ego-nets per edge.
3. **Feature engineering:**

   * Topology/role stats (in/out-degree, unique counterparties, entropy), temporal gaps/burstiness, short cycles, fan ratios, path-continuation likelihood.
   * **Aux categorical encoding** (credit-card/reinvestment) via NLP embeddinga.
4. **Scaling & regularization:** Standardize continuous features (e.g., `sklearn.preprocessing.StandardScaler`), shrinkage on covariance for distance models.
5. **Class imbalance handling:** weighted loss or focal loss; stratified time splits.

✅ **3+ ML Algorithms/Models Identified**

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


✅ **3+ Quantitative Metrics**

* **Detection:** AUROC, **Average Precision (PR-AUC)**, **Recall@FPR=1%** (ops-friendly), F1, **Brier score** + **Expected Calibration Error (ECE)**.
* **Clustering/Representation:** **ARI**, **NMI**, Silhouette (for motif separation).
* **Compute/Latency:** per-window feature time, per-edge scoring time.

✅ **Project Goals**

* **Accuracy & Reliability:**

  * +3–7 **AUROC** points vs. raw-features baseline;
  * **+5–10% Recall@FPR=1%**;
  * **ECE ≤ 0.05** after calibration for trustworthy probabilities.
* **Interpretability:** Every alert includes motif rationale (e.g., “fan-in proximity 0.81 + rapid aggregation”).
* **Ethics & governance:**

  * Minimize investigator burden (**lower FPR**) to reduce unnecessary account holds.
  * Auditability: store motif signatures and SHAP attributions for decisions.
  * **Bias checks:** monitor subgroup FPR/TPR across channels/regions; drop spurious categorical signals if they induce drift or bias.


✅ **Expected Results**

* **Motif separation:** ARI/NMI ≫ random; clear UMAP clusters in distance/signature space.
* **Detection lift:** Adding motif signature (`p[8]`, hard label `c`) to raw features improves **PR-AUC** notably in the low-FPR regime.
* **Calibration:** Post-training isotonic/Platt calibration yields **well-calibrated** risk scores (tight reliability curves).
* **Operational readiness:** Linear-time feature extraction, cached prototypes, and sub-millisecond per-edge scoring for tabular models; GNN used selectively where graph context is essential.
