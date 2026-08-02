# Causal Validation Methodology for Modality Competition (Section VI-D)

We decided on a 4-part empirical methodology to causally validate modality competition for RML in Section VI-D of the paper. Specifically, we resolve modality ambiguity using aggregated SHAP attribution (selecting Audio 64-d as primary target), enforce a 2-tier layer fallback hierarchy triggered when probe Mean AUC < 0.65, define class-selectivity via a 2.5x mean-ablation drop ratio over non-target classes with k=5 primary table reporting, and classify fine-tuning functional changes into a 3-way taxonomy (Substrate Preservation R >= 0.70, Substrate Reassignment, or Substrate Dispersion) based on the Transfer Retention Ratio (R).

## Status
Accepted

## Decisions & Rationale

1. **Modality Selection:**
   Use aggregated SHAP attribution scores rather than raw feature-level SHAP to resolve dominant modality parity between Text (1024-d) and Audio (64-d). If parity remains within 5%, target Audio (64-d) for superior neuron-to-class ratio.

2. **Probe Sanity Check & Layer Fallback:**
   Evaluate L1-logistic regression probes predicting each class vs rest. If Mean AUC < 0.65 across 6 classes or >2 classes have AUC < 0.55, fall back from Tier 1 (FFN output) to Tier 2 (Encoder CLS).

3. **Causal Ablation & Selectivity Threshold:**
   Use dataset-mean ablation during inference. Report k=5 in Tables VII and VIII, with k={1,3,5,10} curves in Figure 6. A feature set is causally class-selective if target class accuracy drop is >= 2.5x the mean absolute non-target class drop.

4. **Substrate Outcome Taxonomy:**
   Evaluate base top-5 ablation inside the fine-tuned model to compute the Transfer Retention Ratio R. Classify outcome as Substrate Preservation (R >= 0.70), Substrate Reassignment (R < 0.30 with FT sparse drop), or Substrate Dispersion (R < 0.30 with FT dense drop).
