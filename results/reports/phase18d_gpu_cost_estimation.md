# Phase 18D: GPU Requirements & Training Cost Estimation

**Generated:** 2026-11-03 19:35:43

## 1. Workload Specifications

- **Training Dataset:** Jud-IPL Binary Outcome Prediction (24,849 documents)
- **Validation Dataset:** 5,335 documents
- **Document Chunking:** 8 chunks × 256 tokens per judgment
- **Epochs:** 3
- **Effective Batch Size:** 16 (micro-batch size 4, accumulation steps 4)

## 2. Hardware Requirements & Estimated Cost

| Environment | Hardware Spec | Est. Throughput | Runtime (3 Epochs) | Estimated Cloud Cost |
|---|---|---|---|---|
| **Cloud GPU (Recommended)** | NVIDIA A10G (24GB VRAM) / A100 | ~12 docs/sec | **~1.7 hours** | **$2.59** |
| **Consumer GPU** | NVIDIA RTX 3090/4090 (24GB) | ~14 docs/sec | **~1.5 hours** | Local hardware |
| **Local CPU (Current)** | Multi-core x86_64 CPU | ~0.4 docs/sec | **~51.8 hours** | Infeasible for real-time turn |

## 3. Deployment Protocol

1. **Baselines Executed:** TF-IDF + Logistic Regression, Linear SVM, and N-gram models are executed locally on CPU.
2. **Transformers Implemented & Smoke Tested:** All 4 architectures verified via forward/backward gradient checks.
3. **Full Training Readiness:** Framework provides PyTorch mixed precision (AMP), gradient accumulation, and early stopping ready for GPU cluster execution.
