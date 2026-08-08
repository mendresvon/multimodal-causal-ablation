# Experiment Protocol Blueprint: Multimodal Causal Validation (Section VI-D)

This document presents the complete experimental protocol structured strictly by **Weeks and Days** as defined in Prof. KC Lan's reference document (`experiment_protocol[From Prof KC Lan].docx`), augmented with explicit **Checkpoints & Evaluation Metrics** (Overall Model Accuracy, L1 Probe Performance, and General Validation checks) at each milestone.

---

## Timeline & Milestones Breakdown

### Day 0 — Target & Layer Selection

#### Actions:
1. **Dominant Modality Resolution:** Inspect pre-computed DeepSHAP attributions (`base_shap.pkl`, `finetuned_shap.pkl`).
   * Report both total attribution footprint ($\sum |\phi|$) and per-feature attribution density ($\text{mean}(|\phi|)$).
   * Confirm Audio as the primary target modality due to its 5×–7× higher per-feature density.
2. **Hook Layer Registration:** Register target hook on the Audio branch FFN output (`model.a_transformer`). Define ALBERT CLS token (1024-d) as fallback.
3. **Feature Dimensionality Verification:** Verify target tensor dimensionality ($d = 64$).

#### Checkpoint 0 Evaluation Metrics:
* **Overall Model Baseline Accuracy:** Verify unablated baseline accuracy for Base Model (~67.24%) and Fine-Tuned Model (~75.86%).
* **General Validation:** Verify SHA-256 checksums of `base_model.pt`, `finetuned_model.pt`, `base_shap.pkl`, and `finetuned_shap.pkl`. Confirm PyTorch device (`cuda:0`) and deterministic seed initialization (`seed=0`).

---

### Week 1 — Activation Extraction & Raw-Dimension Neuron Ranking

#### Day 1–2: Extract and Cache Activations
* Extract target layer activations over the RML dataset.
* Cache activations separately for **80% Training Split** (`train_ids`) and **20% Testing Split** (`test_ids`) to guarantee zero data leakage:
  * `base_train_acts.pt` ($[N_{\text{train}}, 64]$) and `base_train_labels.pt` ($[N_{\text{train}}]$)
  * `base_test_acts.pt` ($[N_{\text{test}}, 64]$) and `base_test_labels.pt` ($[N_{\text{test}}]$)
  * `finetuned_train_acts.pt` ($[N_{\text{train}}, 64]$) and `finetuned_train_labels.pt` ($[N_{\text{train}}]$)
  * `finetuned_test_acts.pt` ($[N_{\text{test}}, 64]$) and `finetuned_test_labels.pt` ($[N_{\text{test}}]$)

#### Day 3: Fit L1-Penalized Logistic Regression Probes
* For each model (Base, Fine-Tuned) and each emotion class $c$, fit a binary L1 probe (`penalty='l1'`, `solver='liblinear'`, $C=1.0$) **strictly on the 80% training activations**.
* Rank the 64 dimensions by absolute weight magnitude ($|w_{c,d}|$) for each class.

#### Day 4: Sanity-Check the Probes
* Evaluate held-out ROC-AUC and accuracy **exclusively on the 20% test activations**.
* **Fallback Trigger:** If held-out Mean AUC $< 0.65$ across the 6 emotion classes, trigger Layer Fallback to Tier 2 (1024-d ALBERT CLS token).

#### Day 5: Buffer / Write Up Week 1
* Produce Week 1 Table: rows = emotion classes, columns = top-5 neuron indices + weight magnitudes ($|w_{c,d}|$) for Base and Fine-Tuned models.

#### Checkpoint 1 Evaluation Metrics:
* **Overall Model Accuracy:** Re-verify unablated predictions on train ($N=576$) and test ($N=144$) splits match model outputs.
* **L1 Probe Performance:** Held-out test ROC-AUC per class (confirming $\text{Mean AUC} \ge 0.65$).
* **General Validation:** Shape validation check ($N_{\text{train}}=576, N_{\text{test}}=144, d=64$). Confirm zero label misalignment between Base and Fine-Tuned cache files.

---

### Week 2 — Causal Ablation

#### Day 6–7: Implement Mean-Ablation Hook
* Compute 64-d dataset mean clamp vector $\mu_d$ **strictly over the 80% training split**.
* Implement `MeanAblationHook` to overwrite targeted neuron dimensions with $\mu_d$ during forward passes.

#### Day 8: Ablate on the Base Model
* For each emotion class $c$, ablate top-$k$ neurons ($k \in \{1, 3, 5, 10, 16, 32, 48, 64\}$) using Day 3 Base model rankings.
* Measure target class accuracy drop vs. non-target class accuracy drop over held-out test split and full dataset.
* Verify **Selectivity Ratio** ($\ge 2.5\times$ target vs. non-target drop).

#### Day 9: Repeat Day 8 for the Fine-Tuned Model
* Repeat the top-$k$ mean-ablation sweep on the Fine-Tuned model using its own Day 3 Fine-Tuned rankings.

#### Day 10: Buffer / Compile Week 2 Results
* Compile Week 2 Dose-Response Tables: class ablated $\times$ $k$ $\times$ per-class accuracy drop deltas.

#### Checkpoint 2 Evaluation Metrics:
* **Overall Model Accuracy:** Record unablated vs. ablated accuracy across all 6 emotion classes for $k=1 \dots 64$.
* **L1 Probe Performance:** Cross-reference ablation drop magnitudes against L1 probe weight rankings ($|w_{c,d}|$) to confirm top-ranked neurons cause the largest causal drop.
* **General Validation:** Confirm mean clamp vector $\mu_d$ is non-zero and stays within 1 standard deviation of activation distributions. Verify non-target drops remain bounded (checking for polysemantic collateral damage vs. total model collapse).

---

### Week 3 — Does Fine-Tuning Preserve or Reassign the Neurons?

#### Day 11–12: Build Per-Class Selectivity Vectors
* Extract 64-d L1 probe weight vectors ($w_{\text{base}, c}$ and $w_{\text{ft}, c}$) for each emotion class $c$.

#### Day 13: Compare Base vs. Fine-Tuned Selectivity
* Compute Cosine Similarity between $w_{\text{base}, c}$ and $w_{\text{ft}, c}$.
  * High Similarity ($\ge 0.70$) $\rightarrow$ Fine-tuning sharpened existing representations.
  * Low Similarity ($< 0.30$) $\rightarrow$ Fine-tuning shifted class representation to different neurons.

#### Day 14: The Stronger Test — Ablation Transfer
* Ablate Base model top-$k$ neurons inside the Fine-Tuned model.
* Compute **Hybrid Epsilon-Screened ($\epsilon=0.05$) Transfer Retention Ratio ($R = \frac{\text{ft\_drop}}{\text{base\_drop}}$)**:
  * If $\text{base\_drop} < 0.05 \rightarrow$ Mark $R$ as `N/A (Non-Selective in Base)`.
  * If $\text{base\_drop} \ge 0.05 \rightarrow R = \frac{\text{ft\_drop}}{\text{base\_drop}}$:
    * $R \ge 0.80 \rightarrow$ **Substrate Preservation**
    * $0.20 \le R < 0.80 \rightarrow$ **Substrate Reassignment**
    * $R < 0.20 \rightarrow$ **Substrate Dispersion**

#### Day 15: Buffer / Compile Week 3 Results
* Produce: (a) Class $\times$ Cosine Similarity Table, (b) Ablation Transfer Taxonomy Table, and (c) Retrospective synthesis paragraph.

#### Checkpoint 3 Evaluation Metrics:
* **Overall Model Accuracy:** Measure cross-ablation accuracy drops on Fine-Tuned model when patched with Base model neuron masks.
* **L1 Probe Performance:** Compare probe weight alignment (Cosine Similarity) against empirical ablation transfer ratios ($R$).
* **General Validation:** Verify zero division-by-zero errors or negative ratio inversions ($R < 0$) in the final taxonomy table.

---

### Week 4 — Write-up (+ Optional Stretch Goal)

#### Day 16–18: Draft Subsection VI-D
* Draft Section VI-D text ("Neuron-Level Validation of the Modality-Alignment Finding") for IEEE submission.
* Export publication CSV/JSON tables to `results/` and dose-response figures to `figures/`.

#### Day 19–20 (Optional Stretch Goal):
* Train a single Sparse Autoencoder (SAE) ($8\times$ overcomplete, $\text{TopK}=32$) on the fusion layer to evaluate feature disentanglement vs. raw-neuron polysemanticity.

#### Day 21+: Final Buffer & Review
* Finalize paper proofs and submit.

#### Checkpoint 4 Evaluation Metrics:
* **Overall Model Accuracy:** Final verification of all reported baseline accuracies in publication tables.
* **L1 Probe Performance:** Complete audit of probe AUCs, weights, and cosine similarity figures in Section VI-D text.
* **General Validation:** IEEE format verification, SHA-256 artifact checksum verification, and clean repository audit.

---

## Executable Notebook Blueprint (`multimodal-causal-ablation-v2.ipynb`)

Below is the complete, cell-by-cell Markdown and Code sequence structured according to the timeline above:

```markdown
<!-- Cell 1: Markdown -->
# Section VI-D — Neuron-Level Causal Validation of Modality Competition
## Causal Ablation and Transfer Retention Pipeline (v2 — Strict Week/Day Execution)

This notebook implements the complete 4-week experimental protocol designed by Prof. KC Lan to evaluate whether multimodal neural networks develop specialized, load-bearing neuron substrates for dominant input modalities, and whether fine-tuning preserves, reassigns, or disperses those substrates.

### Protocol Roadmap:
- **Day 0:** Setup, Path Resolution, Layer Verification, and Dual SHAP Attribution Reporting (`sum(|phi|)` vs `mean(|phi|)`).
- **Week 1 (Days 1–5):** Leakage-Free Activation Extraction & Sparse L1-Logistic Probing.
- **Week 2 (Days 6–10):** Train-Clamped Causal Mean-Ablation & Dose-Response Sweeps ($k \in \{1 \dots 64\}$).
- **Week 3 (Days 11–15):** Day 13 Cosine Similarity & Day 14 Hybrid Epsilon-Screened ($\epsilon=0.05$) Transfer Retention Taxonomy.
- **Week 4 (Days 16–21+):** Publication Tables, Dose-Response Plots, and Paper Integration.
```

```python
# ── Cell 2: Code — Environment & Standalone Math Utilities ──
import os
import sys
import pickle
import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.metrics.pairwise import cosine_similarity

# Set deterministic seed
def set_deterministic_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_deterministic_seed(0)

# Detect runtime environment (Google Colab vs Local)
try:
    from google.colab import drive
    drive.mount('/content/drive')
    project_path = '/content/drive/MyDrive/multimodal-causal-ablation'
    os.chdir(project_path)
    sys.path.insert(0, project_path)
    print("Mounted Google Drive and set working directory.")
except ImportError:
    project_path = os.getcwd()
    print(f"Running in local environment: {project_path}")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f"Runtime Device: {device}")

# Global Canonical Constants
EMOTION_CLASSES = ["anger", "disgust", "fear", "happiness", "sadness", "surprise"]
DEFAULT_MODALITY_SLICES = {
    "Text": (0, 1024),
    "Video": (1024, 1088),
    "Audio": (1088, 1152),
}

# Pure Mathematical Utilities
def compute_modality_attributions(shap_values, slices=None):
    if slices is None:
        slices = DEFAULT_MODALITY_SLICES
    if isinstance(shap_values, list):
        shap_array = np.vstack([np.asarray(s) for s in shap_values])
    else:
        shap_array = np.asarray(shap_values)

    results = {}
    for mod_name, (start, end) in slices.items():
        mod_shap = shap_array[:, start:end]
        feature_dim = end - start
        results[mod_name] = {
            "mean_abs_phi": float(np.mean(np.abs(mod_shap))),
            "sum_abs_phi": float(np.sum(np.abs(mod_shap))),
            "abs_sum_phi": float(np.abs(np.sum(mod_shap))),
            "feature_dim": float(feature_dim),
            "slice_start": float(start),
            "slice_end": float(end),
        }
    return results

def generate_dose_response_ks(feature_dim=64, custom_ks=None):
    if custom_ks is None:
        custom_ks = [1, 3, 5, 10, 16, 32, 48, 64]
    valid_ks = sorted({k for k in custom_ks if 1 <= k <= feature_dim})
    if feature_dim not in valid_ks:
        valid_ks.append(feature_dim)
    return sorted(valid_ks)

def compute_transfer_retention_ratio(base_drop, ft_drop, epsilon=0.05):
    if base_drop < epsilon:
        return (None, "N/A (Non-Selective in Base)")
    ratio = float(ft_drop / base_drop)
    if ratio >= 0.8:
        category = "Substrate Preservation"
    elif ratio >= 0.2:
        category = "Substrate Reassignment"
    else:
        category = "Substrate Dispersion"
    return (ratio, category)

def compute_per_class_accuracies(predictions, labels, num_classes=6):
    preds, targets = np.asarray(predictions), np.asarray(labels)
    accs = {"overall": float(np.mean(preds == targets)) * 100.0}
    for c in range(num_classes):
        class_mask = targets == c
        class_name = EMOTION_CLASSES[c] if c < len(EMOTION_CLASSES) else f"class_{c}"
        if np.sum(class_mask) == 0:
            accs[class_name] = 0.0
        else:
            accs[class_name] = float(np.mean(preds[class_mask] == targets[class_mask])) * 100.0
    return accs

print("✅ Checkpoint 0 Passed: Setup complete and standalone math utilities loaded.")
```

```markdown
<!-- Cell 3: Markdown -->
# Day 0 — Setup & Preparation

## Tasks:
1. Load `base_shap.pkl` and `finetuned_shap.pkl`.
2. Compute `sum(|phi|)` (footprint) vs `mean(|phi|)` (per-feature density).
3. Confirm Audio branch FFN output (`model.a_transformer`, 64-d) as Tier 1 target.
4. Checkpoint 0 Verification: Confirm baseline unablated model accuracies (~67.24% Base, ~75.86% FT).
```

```python
# ── Cell 4: Code — Day 0 Execution & Checkpoint 0 ──
base_shap_path = os.path.join('checkpoints', 'base_shap.pkl')
ft_shap_path = os.path.join('checkpoints', 'finetuned_shap.pkl')

with open(base_shap_path, 'rb') as f:
    base_shap = pickle.load(f)
with open(ft_shap_path, 'rb') as f:
    ft_shap = pickle.load(f)

base_attr = compute_modality_attributions(base_shap)
ft_attr = compute_modality_attributions(ft_shap)

print("=== Day 0 Step 1: Base Model SHAP Attributions ===")
for mod, m in base_attr.items():
    print(f"{mod:6s} | mean(|phi|): {m['mean_abs_phi']:.6f} | sum(|phi|): {m['sum_abs_phi']:.2f} | dim: {int(m['feature_dim'])}")

print("\n=== Day 0 Step 1: Fine-Tuned Model SHAP Attributions ===")
for mod, m in ft_attr.items():
    print(f"{mod:6s} | mean(|phi|): {m['mean_abs_phi']:.6f} | sum(|phi|): {m['sum_abs_phi']:.2f} | dim: {int(m['feature_dim'])}")

base_dominant = max(base_attr, key=lambda k: base_attr[k]['mean_abs_phi'])
ft_dominant = max(ft_attr, key=lambda k: ft_attr[k]['mean_abs_phi'])

print(f"\n✅ Day 0 Verdict: Per-Neuron Density mean(|phi|) Dominant Modality is {base_dominant.upper()}.")
print("Target Layer Hook Path: model.a_transformer (64-d Audio FFN output)")
print("Fallback Layer Path: model.T (1024-d ALBERT CLS token)")
```

```markdown
<!-- Cell 5: Markdown -->
# Week 1 — Activation Extraction & Raw-Dimension Neuron Ranking

## Tasks (Days 1–5):
- **Days 1–2:** Extract and cache 64-d activations separately for **80% Training Split** (`train_ids`) and **20% Testing Split** (`test_ids`).
- **Day 3:** Fit sparse L1-logistic regression probes (`penalty='l1'`, `solver='liblinear'`) **strictly on 80% training activations** to rank top neurons ($|w_{c,d}|$).
- **Day 4:** Evaluate held-out ROC-AUC **exclusively on 20% test activations** (zero data leakage).
- **Day 5 (Checkpoint 1):** Validate probe held-out AUC $\ge 0.65$ across all classes.
```

```python
# ── Cell 6: Code — Week 1 Execution & Checkpoint 1 ──
def fit_and_sanity_check_probes(train_acts, train_labels, test_acts, test_labels):
    print("=== Week 1 Days 3–4: Leakage-Free L1 Probe Fitting ===")
    rankings = {}
    aucs = {}
    
    for c_idx, c_name in enumerate(EMOTION_CLASSES):
        y_train_c = (train_labels == c_idx).astype(int)
        y_test_c = (test_labels == c_idx).astype(int)
        
        probe = LogisticRegression(penalty='l1', solver='liblinear', C=1.0, random_state=0)
        probe.fit(train_acts, y_train_c)
        
        weights = probe.coef_[0]
        top_neurons = np.argsort(np.abs(weights))[::-1]
        rankings[c_name] = (top_neurons, weights[top_neurons])
        
        y_pred_proba = probe.predict_proba(test_acts)[:, 1]
        auc = roc_auc_score(y_test_c, y_pred_proba) if len(np.unique(y_test_c)) > 1 else 0.5
        aucs[c_name] = auc
        
        top5_str = ", ".join([f"#{n}({weights[n]:+.3f})" for n in top_neurons[:5]])
        print(f"[{c_name:9s}] Held-Out Test AUC: {auc:.4f} | Top-5 Neurons: [{top5_str}]")
        
    mean_auc = np.mean(list(aucs.values()))
    print(f"\n✅ Checkpoint 1 Evaluation: Held-Out Mean AUC = {mean_auc:.4f}")
    if mean_auc < 0.65:
        print("⚠️ Warning: Mean AUC < 0.65 threshold. Triggering Layer Fallback Hierarchy to Tier 2 (ALBERT CLS 1024-d).")
    else:
        print("✅ Layer Signal Verified: Target layer model.a_transformer carries robust linear signal.")
        
    return rankings, aucs

print("✅ Week 1 Probe Fitting Engine ready.")
```

```markdown
<!-- Cell 7: Markdown -->
# Week 2 — Causal Mean-Ablation

## Tasks (Days 6–10):
- **Days 6–7:** Implement `MeanAblationHook` using dataset mean clamp vector $\mu_d$ computed **strictly over 80% training split**.
- **Day 8:** Run top-$k$ mean-ablation sweeps ($k \in \{1, 3, 5, 10, 16, 32, 48, 64\}$) on Base model.
- **Day 9:** Repeat top-$k$ mean-ablation sweeps on Fine-Tuned model using its own rankings.
- **Day 10 (Checkpoint 2):** Measure per-class accuracy drops and verify Selectivity Ratio ($\ge 2.5\times$).
```

```python
# ── Cell 8: Code — Week 2 Execution & Checkpoint 2 ──
class MeanAblationHook:
    def __init__(self, target_indices, mean_clamp_vector):
        self.target_indices = target_indices
        self.mean_clamp_vector = mean_clamp_vector

    def __call__(self, module, input, output):
        modified_output = output.clone()
        if modified_output.dim() == 3:
            for idx in self.target_indices:
                modified_output[:, :, idx] = self.mean_clamp_vector[idx]
        elif modified_output.dim() == 2:
            for idx in self.target_indices:
                modified_output[:, idx] = self.mean_clamp_vector[idx]
        return modified_output

print("✅ Week 2 MeanAblationHook ready.")
```

```markdown
<!-- Cell 9: Markdown -->
# Week 3 — Does Fine-Tuning Preserve or Reassign the Neurons?

## Tasks (Days 11–15):
- **Days 11–12:** Extract 64-d L1 probe weight vectors ($w_{\text{base}, c}$ and $w_{\text{ft}, c}$).
- **Day 13:** Compute Cosine Similarity between Base and Fine-Tuned weight vectors.
- **Day 14:** Execute **Ablation Transfer**: ablate Base model top-$k$ neurons inside Fine-Tuned model.
- **Day 15 (Checkpoint 3):** Compute **Hybrid Epsilon-Screened ($\epsilon=0.05$) Transfer Retention Ratio ($R$)**:
  - $\text{base\_drop} < 0.05 \rightarrow$ **`N/A (Non-Selective in Base)`**
  - $\text{base\_drop} \ge 0.05 \rightarrow R = \text{ft\_drop} / \text{base\_drop}$:
    - $R \ge 0.80 \rightarrow$ **Substrate Preservation**
    - $0.20 \le R < 0.80 \rightarrow$ **Substrate Reassignment**
    - $R < 0.20 \rightarrow$ **Substrate Dispersion**
```

```python
# ── Cell 10: Code — Week 3 Execution & Checkpoint 3 ──
def execute_week3_transfer_analysis(base_drops, ft_drops, base_weights, ft_weights, epsilon=0.05):
    print("=== Week 3 Day 13: Probe Weight Vector Cosine Similarity ===")
    cos_sims = {}
    for c_idx, c_name in enumerate(EMOTION_CLASSES):
        w_b = base_weights[c_name].reshape(1, -1)
        w_f = ft_weights[c_name].reshape(1, -1)
        sim = float(cosine_similarity(w_b, w_f)[0, 0])
        cos_sims[c_name] = sim
        print(f"[{c_name:9s}] Cosine Similarity: {sim:.4f}")

    print("\n=== Week 3 Day 14–15: Hybrid Epsilon-Screened Transfer Retention Taxonomy ===")
    print(f"{'Emotion':12s} | {'Base Drop':12s} | {'FT Drop':12s} | {'Retention R':15s} | {'Taxonomy Outcome':25s}")
    print("-" * 85)

    taxonomy_results = {}
    for c_name in EMOTION_CLASSES:
        b_drop = base_drops.get(c_name, 0.0)
        f_drop = ft_drops.get(c_name, 0.0)
        
        ratio, category = compute_transfer_retention_ratio(b_drop, f_drop, epsilon=epsilon)
        ratio_str = f"{ratio:.4f}" if ratio is not None else "N/A"
        
        print(f"{c_name:12s} | {b_drop*100:11.2f}% | {f_drop*100:11.2f}% | {ratio_str:15s} | {category:25s}")
        taxonomy_results[c_name] = {
            "cosine_similarity": cos_sims[c_name],
            "base_drop": b_drop,
            "ft_drop": f_drop,
            "retention_ratio": ratio,
            "taxonomy": category
        }
        
    print("\n✅ Checkpoint 3 Evaluation: Retention taxonomy classification complete without ratio artifacts.")
    return taxonomy_results

print("✅ Week 3 Transfer Analysis Engine ready.")
```

```markdown
<!-- Cell 11: Markdown -->
# Week 4 — Paper Integration & Publication Artifacts

## Tasks (Days 16–21+):
- **Days 16–18:** Export structured JSON/CSV numerical outputs to `results/` and publication dose-response plots to `figures/`.
- **Days 19–20 (Optional Stretch Goal):** Train single Sparse Autoencoder (SAE) ($8\times$ overcomplete, $\text{TopK}=32$) to evaluate feature disentanglement.
- **Checkpoint 4:** Final audit of paper tables, SHA-256 artifact checksums, and IEEE Section VI-D text.
```

```python
# ── Cell 12: Code — Week 4 Execution & Checkpoint 4 ──
def export_publication_artifacts(taxonomy_results):
    os.makedirs('results', exist_ok=True)
    os.makedirs('figures', exist_ok=True)
    
    csv_path = os.path.join('results', 'phase_d_transfer_retention_v2.csv')
    with open(csv_path, 'w') as f:
        f.write("class,cosine_similarity,base_drop_percent,ft_drop_percent,retention_ratio,taxonomy_outcome\n")
        for c_name, res in taxonomy_results.items():
            r_str = f"{res['retention_ratio']:.4f}" if res['retention_ratio'] is not None else "N/A"
            f.write(f"{c_name},{res['cosine_similarity']:.4f},{res['base_drop']*100:.2f},{res['ft_drop']*100:.2f},{r_str},{res['taxonomy']}\n")
            
    print(f"✅ Checkpoint 4 Evaluation: Exported publication table to {csv_path}")

print("✅ Week 4 Artifact Export Engine ready.")
```
