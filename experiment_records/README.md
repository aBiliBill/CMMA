# Selected experiment records

[Download the experiment record archive](CMMA_实验记录选集.zip).

This release contains selected complete archived runs for the main experiments, two University-1652 ablation studies, and both cross-dataset directions.

| Experiment | Released seed | Relationship to the current manuscript |
|---|---:|---|
| University-1652 main experiment and ablations | 4 | Same archived runs used for the current University results |
| SUES-200 within-dataset, four heights | 4 | Alternative archived repeated runs; not the exact runs underlying the current SUES comparison tables |
| University-1652 to SUES-200 | 1 | Same source run and evaluations used for this transfer experiment |
| SUES-200 to University-1652 | 1 | Same four source-height models and evaluations used for this transfer experiment |

Run identity matters: the released SUES seed-4 repetition is separate from the runs represented in the current manuscript. The archive is a selected-run release, not a claim of exact agreement with every paper value or a multi-seed performance average. Its README and selection.json describe the post hoc selection criteria, seeds, configurations and source hashes.

The archive contains 13 training runs, 9,877 training-step rows, 36 evaluations and 418,200 per-query records. All steps and queries, including failures, are preserved for the included runs. No weights, parameter tensors, optimizer states, feature caches or dataset images are included.

After extracting, run python verify_records.py inside CMMA_Selected_Experiment_Records. The standard-library script verifies file integrity and independently recomputes the recorded metrics; it does not rerun model inference.

Archive SHA256: `026179936c76fb305478cedea8d7615d0848f390844da3a439161731e0309099`
