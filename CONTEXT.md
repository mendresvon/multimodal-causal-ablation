# Multimodal Emotion Causal Validation

Domain model and ubiquitous language for neuron-level causal validation of modality competition in multimodal emotion recognition.

## Language

**Dominant Modality**:
The single input modality (audio, visual, or text) that contributes the highest aggregated DeepSHAP attribution score for a given model and dataset combination.
_Distinguish from_: Primary channel, main signal, winning modality

**Causal Ablation**:
An experimental intervention during inference where specific neuron activations are overwritten to measure their direct causal effect on classification performance.
_Distinguish from_: Zeroing out (sets activations to zero, pushing vectors off-distribution), feature masking, pruning

**Mean Ablation**:
A causal ablation technique where targeted neuron activations are replaced by their precomputed dataset-level mean values to keep the activation vector on-distribution.
_Distinguish from_: Zero-ablation (sets activations to zero rather than dataset mean), zero-masking

**Neuron Selectivity Vector**:
A per-class d-dimensional vector representing the difference between the mean activation of samples belonging to a target emotion class and the mean activation of samples not belonging to that class.
_Distinguish from_: Class feature vector (not class-contrastive), activation signature

**Probe Linear Signal Validation**:
A sanity check performed in Phase B by evaluating L1-logistic regression probes predicting each emotion class. Layer fallback (Tier 1 -> Tier 2 -> Tier 3) triggers if Mean AUC across 6 classes < 0.65 or >2 classes have AUC < 0.55.
_Distinguish from_: Probe check (too vague), classifier test (implies a standalone evaluation)

**Causally Class-Selective Feature Set**:
A set of top-k neurons whose mean-ablation causes a target class accuracy drop at least 2.5x greater than the mean absolute accuracy drop across non-target classes.
_Distinguish from_: Class neuron group (not necessarily causal), target feature set

**Transfer Retention Ratio (R)**:
The ratio of the accuracy drop when base-model top neurons are ablated inside the fine-tuned model compared to when ablated inside the base model (R = ft_drop / base_drop). Under ADR 0003, R is evaluated only when base_drop >= epsilon (0.05); otherwise marked as N/A (Non-Selective in Base).
_Distinguish from_: Cross-ablation score, transfer ratio (too generic)

**Substrate Preservation**:
Outcome where R >= 0.80, proving fine-tuning preserves and sharpens the base model's physical load-bearing neurons.
_Distinguish from_: Substrate retention, feature preservation (does not imply sharpening)

**Substrate Reassignment**:
Outcome where 0.20 <= R < 0.80, proving class representation partially shifted to different units within the modality.
_Distinguish from_: Neuron drift (implies gradual change), feature relocation

**Substrate Dispersion**:
Outcome where R < 0.20, proving representation lost reliance on base substrate neurons.
_Distinguish from_: Feature blurring, representation decay (implies degradation rather than redistribution)

**Polysemantic Collateral Damage**:
Accuracy drops observed on non-target emotion classes when ablating top-k neurons of a target class, caused by individual raw neurons representing multiple features in superposition.
_Distinguish from_: Off-target degradation, generic model failure

