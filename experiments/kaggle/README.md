# Kaggle Experiments

Current Kaggle kernels use one directory per Kaggle script. Each directory keeps:

- `script.py`: the runnable Kaggle script
- `kernel-metadata.json`: the Kaggle kernel metadata
- `output/*.json`: compact result summaries committed to git

Large adapter weights, logs, and generated Kaggle HTML are intentionally ignored.

| directory | purpose |
|-----------|---------|
| [`kaggle_syspref3/`](kaggle_syspref3/) | adversarial system-prompt preference controls for Qwen3 risk adapters |
| [`kaggle_existing_judge_drift/`](kaggle_existing_judge_drift/) | existing-organism judge-decomposition test |
| [`kaggle_existing_value_judge_drift/`](kaggle_existing_value_judge_drift/) | value-relevant judge-decomposition variant |
| [`kaggle_bsa_dataset_organisms/`](kaggle_bsa_dataset_organisms/) | broad BSA organism training across risk/time/apples |
| [`kaggle_bsa_risk_stronger/`](kaggle_bsa_risk_stronger/) | stronger BSA risk-only organism training |
| [`kaggle_bsa_risk_safe_controls/`](kaggle_bsa_risk_safe_controls/) | robustness controls for `risk_safe_multi` |
