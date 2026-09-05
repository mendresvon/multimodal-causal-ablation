# Methodological assessment: probe ranking versus causal neuron screening

## Bottom line

For the stated estimand, namely which of 64 audio-CLS coordinates are load-bearing for the frozen multimodal model's anger behavior, single-neuron intervention screening is the more construct-valid primary method. It directly measures a controlled effect of changing an internal variable on the model output. An L1 logistic probe answers a different question: which coordinates allow a separately trained linear classifier to decode anger labels. Probe weights are neither interventions nor parameters of the original model, so they should not be interpreted as causal necessity scores.

Approach B is therefore better than Approach A for selecting candidates for a later necessity test, but it is not automatically an airtight necessity test. Its result is conditional on the intervention, metric, population of examples, and model checkpoint. Screening and confirmation must use independent data. Single-coordinate screening also misses redundancy and can misstate synergy. The final claim should be "causally important under the prespecified intervention in this frozen model," not "the anger neuron" or an unrestricted causal claim.

The reported audit is strong evidence that probe magnitude is a poor *surrogate* for the causal estimand in this repository. The negative or zero rank correlations in 11/12 model-class cases directly contradict monotonic correspondence between absolute probe weight and ablation damage. However, damage-selected top-5 will necessarily look best if selection and evaluation use the same observations. The 16%-73% comparison becomes confirmatory evidence only when the rankings are frozen before evaluation on independent held-out examples (or inside an outer cross-validation loop).

The repository table gives ratios from 0.1585 to 0.7306 (median 0.4608), a median top-union Spearman correlation of approximately -0.35, and only 1.25 overlapping coordinates on average between the two top-5 lists. Those are large descriptive disagreements. The CSV itself does not contain inferential statistics, however, and a rank correlation restricted to a data-selected top-16 union can be selection-biased. The confirmatory comparison should compute the ranking on discovery data, then estimate joint top-5 damage and all-64 rank agreement on a locked test set.

## What each approach identifies

Let `z_c(x)` be the original model's class-c logit, `h_j(x)` audio-CLS coordinate j, and `b_j` a prespecified replacement rule. A class-c single-neuron effect can be defined on held-out examples as

`D_jc = E[m_c(x) - m_c(x; do(h_j := b_j)) | Y=c]`,

where `m_c` is a continuous target-vs-rest logit margin. This is a controlled interventional effect in the deterministic computation graph. It tests whether the model's output depends on that coordinate under that particular intervention. Neural causal-mediation and interchange-intervention work explicitly uses internal interventions for this purpose (Vig et al., 2020; Geiger et al., 2022).

By contrast, an L1 probe estimates parameters of a new predictor `P_probe(Y=c | h)`. It establishes linear decodability or class association. It does not establish that the original classifier reads, needs, or uses a coordinate. Elazar et al. explicitly distinguish information extractable by a probe from information behaviorally used by the model. Haufe et al. show more generally that weights of discriminative backward models are not source/activation patterns and can be nonzero for variables statistically independent of the signal source. Lasso support recovery additionally depends on covariance conditions, and correlated predictors can be selected unstably or arbitrarily (Zhao and Yu, 2006; Zou and Hastie, 2005).

Absolute probe weights create two additional issues: coefficient magnitudes depend on feature scale unless activations are standardized using training statistics, and `|w_j|` conflates anger-promoting (`w_j > 0`) with anger-suppressing (`w_j < 0`) directions. For necessity, rank signed loss of target evidence, not absolute change. Report suppressive neurons separately.

## Repository-specific simplification: the downstream map is linear

In this repository, the intervention locus has an unusually useful structure. `a_cls` is fed directly to `a_out = nn.Linear(64, 6)`, and the three modality-logit vectors are combined by `weighted_fusion = nn.Linear(3, 1, bias=False)`. If the audio fusion weight is `alpha_a`, then clamping coordinate j to its training mean changes class-c logit by exactly

`delta z_c(x,j) = alpha_a * W_audio[c,j] * (h_j(x) - mean_train(h_j))`.

This equation explains why an external one-vs-rest probe coefficient need not track damage. Causal impact depends on the model's own six-row audio readout, the audio fusion coefficient, the example's displacement from the clamp value, the competing class logits, and the other modalities' logits. The probe optimizes none of those quantities.

It also sharpens the interaction analysis. At this exact site, multi-coordinate effects are exactly additive in **logit space**. There is no hidden downstream neuron-neuron synergy before the logits. Non-additivity observed in softmax probability, loss, or argmax accuracy can arise from the softmax/decision metric and saturation, not from an interacting audio-CLS circuit. Therefore:

- use the algebraic logit contribution as a transparent audit of the intervention code;
- retain direct forward ablation as the primary behavioral measurement, since it captures multiclass competition in probability/margin space;
- jointly test top-k sets because the reported behavioral metric is nonlinear;
- describe departures from additivity as output-metric interactions unless the intervention is moved to an earlier nonlinear layer.

Redundancy is still meaningful as distributed support: many coordinates may make substitutable or individually small contributions. But at this locus it should not be described as a nonlinear downstream compensation mechanism. A forward-pass equality check between analytic and hooked logit deltas would be a strong falsification test.

## Strengths and weaknesses

### L1 logistic probe

Strengths:

- Cheap, familiar, and useful for testing whether class information is linearly decodable.
- A sparse solution supplies a compact hypothesis set for subsequent intervention.
- Cross-validated predictive performance and control tasks can quantify whether a probe is learning a reproducible association rather than only memorizing labels (Hewitt and Liang, 2019).

Weaknesses:

- Correlational and model-external: its coefficient is not a causal effect and not the original model's readout weight.
- Conditional coefficients can be distorted by multicollinearity, suppressor variables, feature scaling, regularization strength, class imbalance, and one-vs-rest construction.
- L1 often chooses one representative from a correlated group. This can make a redundant causal feature receive zero weight or make a high-weight proxy causally inert.
- A linear probe ignores nonlinear downstream use and neuron interactions.
- A high-capacity or poorly controlled probe can extract information that is not naturally used by the network. Probe selectivity/control-task methodology addresses probe memorization, not causal usage.

### Single-neuron intervention screening

Strengths:

- Directly targets dependence of the frozen model's behavior on a model component.
- Includes the actual downstream nonlinearities and multimodal fusion computation.
- With only 64 coordinates, exhaustive single-coordinate evaluation is computationally straightforward.
- Class-conditional effects can reveal class-specific dependence that global accuracy conceals; prior unit-ablation work found that individual-unit ablation may have little effect globally but substantial effects for particular classes (Zhou et al., 2018).
- It is aligned with established causal-mediation, activation-patching, and neuron-lesion methods (Vig et al., 2020; Ghorbani and Zou, 2020; Zhang and Nanda, 2024).

Weaknesses:

- The estimand changes with the baseline/replacement rule. An intervention can both delete information and inject or "spoof" abnormal information into downstream computation (Li and Janson, 2024).
- Replacing one coordinate independently generally breaks dependencies among coordinates and can create hidden states not produced by natural inputs.
- Leave-one-neuron-out effects are conditional on all other neurons being intact. They can be near zero under redundancy and non-additive under synergy.
- Ranking by target effect alone may identify a generic bottleneck or destabilizing coordinate rather than a class-selective one.
- Screening 64 neurons and reporting the winners on the same data creates winner's-curse/selection bias.
- Individual coordinates are basis-dependent. Fine-tuning can rotate or redistribute a representation while preserving its function, so same-index persistence is only one definition of survival.

## Airtight experimental design

### 1. Predeclare causal estimands and intervention site

- Freeze the model and intervene at a precisely named tensor, preferably the audio CLS vector immediately consumed by fusion/classification. If the intervention occurs before LayerNorm, changing one entry can alter every normalized output coordinate; either intervene after the relevant normalization or explicitly describe the broader intervention.
- Verify with hooks/unit tests that only the intended coordinate changes at the intervention site and that every downstream use of that coordinate is affected.
- Primary target-evidence metric: a continuous logit margin, for example `m_c = z_c - logmeanexp(z_-c)` or `z_c - mean(z_-c)`. Probability is a useful secondary outcome, but softmax coupling and saturation can make probability changes misleading. Accuracy/recall are thresholded secondary behavioral endpoints. Zhang and Nanda (2024) show that metric and corruption choices can materially change localization results and give reasons to prefer logit differences in many settings.
- Define a positive damage as an ablation-induced fall in target margin or rise in target loss. Do not rank by absolute damage if the claim is necessity. Preserve negative effects as evidence of suppressive neurons.

### 2. Separate importance, selectivity, and generic disruption

Report at least three quantities for every neuron:

1. Target necessity: mean paired decrease in class-c margin on true class-c examples; also report change in class-c recall and cross-entropy.
2. Off-target disruption: macro-average change in each other class's own correct-class margin/loss on its true examples.
3. Selective effect: a prespecified contrast such as target necessity minus macro-average off-target disruption. Standardize within class if scale differs, and report both components rather than only the contrast.

Also report the change in the target logit on non-target examples and the target false-positive rate. This distinguishes a coordinate that supplies class-c evidence generally from one needed only for correct class-c recognition. Use logits/margins for class comparisons because all probabilities are coupled through the softmax.

Controls should include equal-size random neuron sets (many draws), bottom-ranked sets, probe-ranked sets, activation-selectivity-ranked sets, and neurons matched on activation variance or norm. Report global macro-F1/accuracy, per-class metrics, predictive entropy, and representation/output norms. A whole-audio ablation supplies a useful upper-bound/reference, while visual-stream controls help establish modality specificity.

### 3. Prevent selection bias and leakage

- Compute every baseline statistic, probe scaler, probe fit, and ablation ranking without touching the final test set.
- Use a discovery split to rank all 64 neurons. Use validation data to choose the intervention rule, metric, top-k procedure, and robustness criteria. Lock the top-5 and analysis before one final evaluation on the untouched test split.
- If RML has repeated speakers or clips, split and resample at the independent speaker/subject level, not only the clip level.
- For limited data, use nested, speaker-grouped cross-validation: all neuron selection occurs inside each outer-training fold and effects are measured only in its outer-test fold. This is the direct analogue of nested model selection. Cawley and Talbot (2010) and Varma and Simon (2006) show why evaluating a selected winner on the selection data is optimistically biased.
- Use paired per-example contrasts because intact and ablated predictions share the same examples. Obtain confidence intervals with a speaker-clustered bootstrap or randomization test. Correct confirmatory neuron-by-class tests across 64 neurons, classes, and prespecified model comparisons (for example, Benjamini-Hochberg FDR). Report effect sizes and confidence intervals, not only p-values.
- Quantify ranking stability across bootstrap discovery samples and training seeds (selection frequency, rank correlation, top-5 Jaccard overlap). Treat multiple model seeds/checkpoints as replication, not merely extra independent examples.

### 4. Make the replacement rule a sensitivity analysis

Training-set mean replacement is defensible as a simple total ablation: it removes input-specific variation and avoids test leakage. It is not uniquely neutral, and a vector assembled from coordinatewise means need not be a natural hidden state.

Recommended hierarchy:

- Primary: training-only mean, if it is close to the coordinate's normal support and does not create obvious downstream norm/statistic anomalies.
- Robustness: zero only when zero has architectural semantics (for example, a genuinely centered or gated-off component), otherwise treat it as a stress test.
- Robustness: resample the coordinate from training examples and average effects over several draws. This preserves the marginal coordinate distribution but still breaks joint dependencies.
- Stronger naturalistic control: patch from matched source examples that alter emotion evidence while matching speaker and nuisance variables. This changes the causal question from total ablation to a counterfactual swap and must be labeled accordingly.
- Optional: conditional replacement estimated from training data, `E[h_j | h_-j]`, to reduce off-manifold states. This estimates the unique residual contribution conditional on other neurons, not total importance.
- Optional modern comparator: optimal ablation, which chooses a constant minimizing expected ablated-model loss and is designed to reduce spoofing artifacts (Li and Janson, 2024).

Conclusions should be called robust only if the leading coordinates and held-out joint effects are qualitatively stable across reasonable replacement rules. Monitor whether ablated activations/downstream states fall outside the intact training distribution.

### 5. Test non-additivity, redundancy, and synergy explicitly

Single-neuron damage is the leave-one-out marginal contribution when all other neurons are present. In a general nonlinear network it is not an additive allocation of the top-5 set's effect. In this repository's current audio-CLS intervention, however, class-logit effects are algebraically additive; only nonlinear behavioral outcomes such as probability, loss, and accuracy need an empirical interaction analysis.

- After selecting five neurons, evaluate the joint top-5 ablation and all 31 nonempty subsets of those five on held-out data.
- For each pair, compute an interaction contrast: joint damage minus the sum of the two single damages. It must be zero to numerical precision for each class logit in the present architecture. For probability/loss, positive values indicate super-additive behavioral damage under the chosen sign convention and negative values indicate sub-additivity or saturation. Do not interpret these probability interactions as evidence of a nonlinear interaction inside `a_cls` or `a_out`.
- Compare the observed joint top-5 effect to the sum of singleton effects and to the empirical distribution from many random matched five-neuron sets.
- Add greedy conditional selection as a sensitivity analysis: at each step choose the neuron with the largest incremental joint damage given the already ablated set. This can avoid choosing five redundant neurons, although it can still miss pure synergies whose singleton effects are zero.
- For a coalition-aware ranking of nonlinear probability/loss effects, estimate Neuron Shapley values from random ablation coalitions. Shapley values average a neuron's marginal contribution over contexts and therefore account for output-metric interactions, at increased compute and with continued dependence on the ablation game/baseline. Ghorbani and Zou (2020) propose this specifically for neurons. Shapley-Taylor indices can summarize explicit pair/higher-order interactions (Sundararajan et al., 2020). For the present model's class logits, Shapley adds no information beyond the closed-form additive contribution.

A null single-neuron effect supports only "not individually necessary with all other units intact." It does not establish irrelevance. Conversely, a large effect establishes vulnerability/dependence under that intervention, not that the neuron alone is sufficient or uniquely implements the function.

### 6. Define "survives fine-tuning" prospectively

Use two distinct analyses:

1. Same-index causal persistence: select and lock the base model's top-5 without fine-tuned test outcomes, then ablate those identical indices in the fine-tuned model on held-out paired examples. Test the checkpoint-by-ablation interaction and report each coordinate's effect before and after fine-tuning. This is the clean primary survival test.
2. Functional relocation: independently screen the fine-tuned model inside its own discovery data, then compare held-out effect vectors, top-k overlap, and ranks. This asks whether the function moved to other coordinates.

Compare persistence against a within-checkpoint random-set null, not only a raw fine-tuned/base damage ratio. A fine-tuned model can be more sensitive to every five-coordinate ablation, making all transfer ratios exceed one without selective survival. The repository's accepted ADR 0006 already documents this confound: after comparison with 2,000 random five-coordinate subsets, none of the six classes showed transfer above the random-ablation null. That result should be described as "no evidence of selective same-index causal persistence above generic fine-tuned fragility," not as proof that all representations were erased or reassigned.

Fine-tuning may rotate or redistribute the representation, so coordinate identity is basis-sensitive. Add a subspace-level comparison: estimate any alignment/matching only on training activations, evaluate it on held-out data, and compare the causal effect of the selected base subspace after alignment. CKA can report whole-representation similarity across checkpoints, but is descriptive rather than a causal survival test (Kornblith et al., 2019). Network Dissection's rotation result shows why axis-aligned neuron interpretations are not invariant to a change of basis (Bau et al., 2017).

For modality competition specifically, an audio-neuron effect in a fused model establishes audio dependence, not competition. Competition requires a contrast, such as whether the same neuron's effect changes when vision is present versus absent/degraded, or before versus after multimodal fine-tuning. Prespecify and test the corresponding interaction.

## Recommended role of Approach A

Retain the probe as a complementary representational analysis, not the primary selector for causal claims. A clean paper design is:

- Probe analysis: where is anger linearly decodable, and how stable/selective is that decoding?
- Causal screen: which coordinates most affect the frozen model's anger margin under prespecified interventions?
- Convergence test: do probe and causal rankings agree? The present audit says largely no.
- Confirmatory group test: does the locked causal top-5 cause class-selective held-out damage beyond matched random, probe-ranked, and bottom-ranked controls?
- Persistence test: does the locked base causal set retain its effect after fine-tuning, and if not, is the effect redistributed?

This framing turns the disagreement between the two approaches into a substantive result: decodable class information and behaviorally used class information are empirically dissociated.

## Primary sources

- Vig, J. et al. (2020). *Investigating Gender Bias in Language Models Using Causal Mediation Analysis*. NeurIPS 33, 12388-12401. https://proceedings.neurips.cc/paper/2020/hash/92650b2e92217715fe312e6fa7b90d82-Abstract.html
- Geiger, A. et al. (2022). *Inducing Causal Structure for Interpretable Neural Networks*. ICML, PMLR 162:7324-7338. https://proceedings.mlr.press/v162/geiger22a.html
- Elazar, Y. et al. (2021). *Amnesic Probing: Behavioral Explanation with Amnesic Counterfactuals*. TACL 9:160-175. https://aclanthology.org/2021.tacl-1.10/ (DOI: 10.1162/tacl_a_00359)
- Hewitt, J. and Liang, P. (2019). *Designing and Interpreting Probes with Control Tasks*. EMNLP-IJCNLP, 2733-2743. https://aclanthology.org/D19-1275/ (DOI: 10.18653/v1/D19-1275)
- Haufe, S. et al. (2014). *On the interpretation of weight vectors of linear models in multivariate neuroimaging*. NeuroImage 87:96-110. https://pubmed.ncbi.nlm.nih.gov/24239590/ (DOI: 10.1016/j.neuroimage.2013.10.067)
- Zhao, P. and Yu, B. (2006). *On Model Selection Consistency of Lasso*. JMLR 7:2541-2563. https://www.jmlr.org/papers/v7/zhao06a.html
- Zou, H. and Hastie, T. (2005). *Regularization and Variable Selection via the Elastic Net*. JRSS B 67(2):301-320. https://doi.org/10.1111/j.1467-9868.2005.00503.x
- Morcos, A. S. et al. (2018). *On the Importance of Single Directions for Generalization*. ICLR. https://openreview.net/forum?id=r1iuQjxCZ
- Zhou, B. et al. (2018). *Revisiting the Importance of Individual Units in CNNs via Ablation*. arXiv:1806.02891. https://arxiv.org/abs/1806.02891
- Ghorbani, A. and Zou, J. (2020). *Neuron Shapley: Discovering the Responsible Neurons*. NeurIPS 33. https://proceedings.neurips.cc/paper/2020/hash/41c542dfe6e4fc3deb251d64cf6ed2e4-Abstract.html
- Sundararajan, M., Dhamdhere, K., and Agarwal, A. (2020). *The Shapley Taylor Interaction Index*. ICML, PMLR 119:9259-9268. https://proceedings.mlr.press/v119/sundararajan20a.html
- Zhang, F. and Nanda, N. (2024). *Towards Best Practices of Activation Patching in Language Models: Metrics and Methods*. ICLR. https://proceedings.iclr.cc/paper_files/paper/2024/hash/06a52a54c8ee03cd86771136bc91eb1f-Abstract-Conference.html
- Li, M. and Janson, L. (2024). *Optimal Ablation for Interpretability*. NeurIPS 37. https://proceedings.neurips.cc/paper_files/paper/2024/hash/c55e6792923cc16fd6ed5c3f672420a5-Abstract-Conference.html (DOI: 10.52202/079017-3468)
- Bau, D. et al. (2017). *Network Dissection: Quantifying Interpretability of Deep Visual Representations*. CVPR, 6541-6549. https://openaccess.thecvf.com/content_cvpr_2017/html/Bau_Network_Dissection_Quantifying_CVPR_2017_paper.html (DOI: 10.1109/CVPR.2017.354)
- Kornblith, S. et al. (2019). *Similarity of Neural Network Representations Revisited*. ICML, PMLR 97:3519-3529. https://proceedings.mlr.press/v97/kornblith19a.html
- Cawley, G. C. and Talbot, N. L. C. (2010). *On Over-fitting in Model Selection and Subsequent Selection Bias in Performance Evaluation*. JMLR 11:2079-2107. https://www.jmlr.org/papers/v11/cawley10a.html
- Varma, S. and Simon, R. (2006). *Bias in error estimation when using cross-validation for model selection*. BMC Bioinformatics 7:91. https://doi.org/10.1186/1471-2105-7-91

## Repository evidence consulted

- [`results/table_x_probe_rank_validation.csv`](../../results/table_x_probe_rank_validation.csv): probe-versus-measured top-5 audit for 12 model-class pairs.
- [`Model/Dig-Data_Model-Main/src/models/e2e.py`](../../Model/Dig-Data_Model-Main/src/models/e2e.py): linear `a_out` and linear modality fusion defining the closed-form logit intervention.
- [`docs/adr/0005-day0-dominant-modality-recompute-and-protocol-alignment.md`](../adr/0005-day0-dominant-modality-recompute-and-protocol-alignment.md): exact fast-path and intervention-fidelity decisions.
- [`docs/adr/0006-week3-transfer-null-and-band-withdrawal.md`](../adr/0006-week3-transfer-null-and-band-withdrawal.md): probability-resolution choice, random-ablation transfer null, and withdrawal of uncalibrated transfer bands.
