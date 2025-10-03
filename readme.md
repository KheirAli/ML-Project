# Anti-Money Laundering Motif Detection  
[Proposal Video](https://youtu.be/wMEg9vx8zRo)  

## Introduction  
Money laundering is the act of moving illicit funds to conceal their origin and make them appear legitimate. According to the UNODC, the estimated impact of money laundering is 2–5% of global GDP each year [1]. In the U.S. alone, money laundering incidents in FY2024 were 1095, up 45% since FY2020 [2], [3].  

Detecting these activities is challenging due to their rarity, adaptability, and the massive volume of legitimate transactions. Machine Learning (ML) offers promise, but real-world datasets are scarce due to privacy and regulatory restrictions [4].  

## About the Dataset  
IBM created a synthetic financial dataset (AMLSim) using agent-based simulation [5]. The dataset includes transactions between accounts, amounts, transaction types, timestamps, and account metadata. This enables experimentation with ML methods while protecting privacy.  

## Related Works  
- **Unsupervised methods:** K-Means, DBSCAN, and ensemble methods such as Random Forest, LightGBM, and XGBoost have been applied to anomaly detection in AML tasks [6].  
- **Supervised methods:** Graph Neural Networks (GNNs) and autoencoders have been explored for supervised AML detection.  
- **Gap:** Existing works rarely focus on *motif-level interpretability*, i.e., explaining anomalies through recurring transaction patterns.  

## Problem Definition & Challenge  
We aim to tag every transaction with its proximity to **8 canonical money-laundering motifs** used in AMLSim:  
- Fan-out, Fan-in, Scatter-gather, Gather-scatter, Simple cycle, Random walk, Bipartite, Stack  

### Approach  
- Build a pattern-aware feature vector from ego-networks, amounts, and timing.  
- Compute an **8-way distance signature** to measure similarity to each motif.  
- Use signatures for clustering (unsupervised motif discovery) and supervised detection.  

### Challenges  
- Money laundering is sparse and adaptive.  
- Overlapping patterns and varying transaction scales.  
- Need for interpretable, robust, and time-aware detection.  

## Data Preprocessing Methods  
- **Time-Window Aggregation:** Sliding windows (7–30 days) to capture temporal motifs.  
- **Robust Scaling:** Prevents domination by outliers in transaction values.  
- **Class Balancing (SMOTE):** Synthesizes minority-class (laundering) transactions.  
- **Graph Construction:** Directed time-windowed graphs reveal fan-out/cycle motifs hidden in tabular data.  

## Methods & Why This Works  

### 1. Unsupervised Learning (Motif Discovery)  
- Construct sliding-window graphs.  
- Extract role, temporal, and motif-based features.  
- Compute distance signatures (Mahalanobis distance).  
- Cluster interactions into motif types.  

**Why effective:** Distances reflect *how* interactions resemble motifs, enabling interpretable clustering.  

### 2. Supervised Learning (Laundering Detection with Motif Priors)  
- Input vector: raw features + distance signatures + motif cluster labels.  
- Models: Logistic regression, Gradient Boosted Trees, optional GNNs.  
- Explainability: SHAP values highlight motif proximity and transaction features.  

**Why effective:** Combines anomaly detection with motif interpretability, improving recall while controlling false positives.  

## Deliverables  
- Feature vectors, motif distance signatures, cluster labels.  
- Clustering metrics (ARI/NMI).  
- Detection metrics (AUROC/PR-AUC).  
- Visualizations of motif separation.  
- Explanations for alerts (“near-fan-out + bursty micro-amounts”).  

## Engineering Outline  
```
/prep/        Graph builder + windowing
/features/    Role/temporal/motif extractors
/prototypes/  Fit & store motif prototypes
/distance/    Compute d, p, c per interaction
/models/      Train/eval detectors, calibration, SHAP
/eval/        Clustering metrics & detection metrics
```  

## Expectations  
- **Pattern separation:** Coherent motif clusters with strong ARI/NMI.  
- **Detection lift:** Higher AUROC/PR-AUC with motif features.  
- **Interpretability:** Alerts with motif-based rationales.  
- **Generalization:** Robust performance across varying laundering rates.  

## Gantt Chart  
[Project Timeline](https://gtvault-my.sharepoint.com/:x:/g/personal/akothapalli31_gatech_edu/EZAg5U5raNFDgFohfCuoDvoBeR2CBnpRpG5yfuF6GvvYyQ?e=nCkaWJ)  

## Contribution Table  

| Name              | Proposal Contribution                                                                 |
|-------------------|----------------------------------------------------------------------------------------|
| **Alireza Kheirandish** | Data investigation, graph generation, feature engineering, unsupervised learning, clustering |
| **Akhil Kothapalli**    | Motif feature extraction, unsupervised learning & clustering, metrics gathering               |  

---

## References  
[1] UNODC, “Model Laundering.” [Online]. Available: https://www.unodc.org/unodc/en/money-laundering/overview.html. [Accessed: Sept. 30, 2025].  

[2] United States Sentencing Commission (USSC), “Money Laundering Quick Facts.” [Online]. Available: https://www.ussc.gov/research/quick-facts/money-laundering. [Accessed: Sept. 30, 2025].  

[3] U.S. Sentencing Commission, “Laundering of Monetary Instruments; Engaging in Monetary Transactions in Property Derived from Unlawful Activity.”  

[4] V. K. Potluru, Y. Sun, J. Song, and L. Zhao, “Synthetic Data Applications in Finance,” *arXiv preprint* arXiv:2401.00081, 2023.  

[5] E. Altman, J. Blanuša, L. von Niederhäusern, B. Egressy, A. Anghel, and K. Atasu, “Realistic Synthetic Financial Transactions for Anti-Money Laundering Models,” *arXiv preprint* arXiv:2306.16424, 2023. [Online]. Available: https://arxiv.org/abs/2306.16424  

[6] F. Gómez Mármol and G. Martínez Pérez, “Anti-Money Laundering Recognition through the Gradient Boosting Classifier,” *ResearchGate*, 2021. [Online]. Available: https://www.researchgate.net/publication/354776829_Anti-Money_Laundering_Recognition_through_the_Gradient_Boosting_Classifier  

