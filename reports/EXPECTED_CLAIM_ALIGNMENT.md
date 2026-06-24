# How to align claims after running

Use this wording only after `scripts/make_report.py` generates the final table.

## If MCoT+VG-v2 is best and significant

> MCoT+VG-v2 significantly improves semantic A-OKVQA direct-answer accuracy over the direct LLaVA baseline while also improving deletion faithfulness, indicating that step-conditioned visual evidence can support both answer quality and interpretability.

## If MCoT+VG-v2 is best but not significant

> MCoT+VG-v2 achieves the highest semantic accuracy among the evaluated non-router configurations, although the gain is modest; its main benefit is the addition of step-conditioned grounding and improved deletion faithfulness.

## If MCoT+VG-v2 matches Baseline/VG

> MCoT+VG-v2 preserves answer accuracy while adding step-wise visual grounding. The main contribution is interpretability and faithfulness, not a statistically significant accuracy improvement.

## If MCoT+VG-v2 is lower

> MCoT+VG-v2 exposes an interpretability–accuracy trade-off: visual grounding improves explanation faithfulness but the current reasoning-feedback design does not improve answer accuracy.

## Never claim unless real annotations are available

- Human Interpretability Score
- Step Consistency
- Fleiss' κ
- human-grounded faithfulness
