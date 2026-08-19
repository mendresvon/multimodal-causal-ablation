# ADR 0003: Causal Ablation Pipeline v2 Methodology Remediation & Refinements

We decided on four critical architectural and methodological refinements for the `multimodal-causal-ablation-v2.ipynb` pipeline to eliminate data leakage, resolve mathematical edge-case artifacts, establish publication-grade attribution reporting, and explicitly address representation polysemanticity.

## Status

**Superseded by [ADR 0004](0004-v3-restart-data-provenance-and-fidelity-gate.md).** Retained as history.

Two records were written under the number 0003 on different days; this is the later one. Decision 1 (dual SHAP reporting, `mean(|phi|)` as the selection criterion) survives and is carried forward by ADR 0005. The ε=0.05 Transfer Retention screen is explicitly retracted by ADR 0004 Decision 4, it was hiding division-by-zero and negative-ratio artifacts produced by a broken harness rather than handling a real statistical edge case. Every number computed under this ADR is retracted.

## Decisions & Rationale

1. **Dual SHAP Attribution Reporting & Layer Target Selection:**
   * **Decision:** Report both `sum(|phi|)` (total modality footprint) and `mean(|phi|)` (per-feature attribution density). Use `mean(|phi|)` density as the primary criterion for layer target selection.
   * **Rationale:** Text (1024-d) has a larger total attribution sum due to its 16× dimension count. However, Audio (64-d) features carry 5×–7× higher per-feature attribution density (`mean(|phi|)`), proving that individual Audio neurons are the most densely packed, load-bearing units. This dual reporting acknowledges prior correlational findings while providing a mathematically sound justification for probing the Audio 64-d branch (`model.a_transformer`).

2. **Leakage-Free 80/20 Train/Test Activation & Probing Protocol:**
   * **Decision:** Cache activations separately for the model's 80% Training Split (`train_ids`) and 20% Testing Split (`test_ids`). Fit sparse L1-logistic regression probes (`penalty='l1'`, `solver='liblinear'`) **strictly on 80% training activations**, and evaluate probe ROC-AUC sanity checks **exclusively on 20% held-out test activations**.
   * **Rationale:** Guarantees zero data leakage in probe fitting and weight magnitude ranking ($|w_{c,d}|$), ensuring that probe weights reflect generalizable linear signals rather than memorized test statistics.

3. **Hybrid Epsilon-Screened ($\epsilon=0.05$) Transfer Retention Ratio ($R$) Taxonomy:**
   * **Decision:** Implement a two-tier evaluation rule for the Transfer Retention Ratio ($R = \text{ft\_drop} / \text{base\_drop}$):
     1. **Epsilon Screening:** If $\text{base\_drop} < 0.05$ (5%), classify the class as **`N/A (Non-Selective in Base)`**.
     2. **Selective Taxonomy Mapping ($\text{base\_drop} \ge 0.05$):**
        * $R \ge 0.8 \rightarrow$ **Substrate Preservation**
        * $0.2 \le R < 0.8 \rightarrow$ **Substrate Reassignment**
        * $R < 0.2 \rightarrow$ **Substrate Dispersion**
   * **Rationale:** Classes with negligible Base model drops ($\text{base\_drop} < 5\%$) are not causally load-bearing in the Base model. Screening them out prevents division-by-zero, negative ratio inversions, and tiny-denominator noise artifacts from distorting publication tables.

4. **Polysemanticity Empirical Framing & SAE Motivation:**
   * **Decision:** Frame raw-neuron polysemanticity (superposition) explicitly in the paper:
     1. Use the **Selectivity Ratio** (Target Class Drop vs. Average Non-Target Class Drop) to empirically distinguish class-specific load-bearing roles (selective drops) from polysemantic collateral damage (non-target drops).
     2. Position raw-neuron polysemanticity as the primary theoretical motivation for Sparse Autoencoders (SAEs), framing SAE feature disentanglement as the Week 4 stretch goal / future work.
   * **Rationale:** Polysemanticity is a known reality in deep networks. Directly measuring and discussing it elevates paper quality to publication grade while protecting against reviewer critique regarding raw-neuron superposition.
