# Official Experimental Protocol

## Scope and precondition

This protocol is executable only after official access to FakeAVCeleb and LipSyncTIMIT, or another documented licensed source. A smoke run on synthetic clips validates software only; it is not a benchmark result and must not be reported in the manuscript.

## Fixed protocol

Each dataset is split by `speaker_id` into 70% train, 15% validation, and 15% test identities with seeds 42, 52, and 62. The best checkpoint is selected solely by validation loss. The held-out test identities are evaluated once per seed. All methods receive 16 RGB frames at 112 by 112 pixels and an 80-bin log-mel representation calculated from the same eight-second window.

The primary outcomes are AUROC, AP, EER, severity MAE, and localization mIoU/Precision@K. Efficiency is measured on the same device after warm-up using model parameters, analytical FLOPs, peak memory, and median/p95 latency. Results must report mean plus standard deviation across seeds and must identify the hardware, software versions, input size, batch size, and timing protocol.

## Planned ablations

`fixed_severity` replaces the severity curriculum with a fixed 0.6 recipe. `no_mamba` removes state-space blocks. `dense_attention` replaces sparse top-k selection with all-frame attention. `no_sae` omits the attribution loss and decoder. Every ablation keeps the dataset split, optimizer, update budget, preprocessing, and seed fixed.
