# Security Audit Report: autoresearch

**Audit Date:** 2026-04-28
**Project:** Autonomous pretraining research swarm (LLM training experimentation framework)
**Repository:** https://github.com/micahstubbs/autoresearch (fork of https://github.com/karpathy/autoresearch)
**Auditor:** Claude Code Security Analysis

## Executive Summary

**VERDICT: CLEAN**

This repository is a fork of Andrej Karpathy's well-known `autoresearch` project — a single-GPU LLM pretraining framework derived from `nanochat`. All Python dependencies come from official package registries (PyPI + the official PyTorch CUDA wheel index). The only network operations are HuggingFace dataset downloads (`huggingface.co`) and HuggingFace Hub kernel loading. No malicious code patterns, obfuscation, post-install hooks, or credential exfiltration vectors were detected. Safe to install.

## Project Overview

| Attribute | Value |
|-----------|-------|
| Language | Python 3.10+ |
| Framework | PyTorch (single-GPU CUDA) |
| Purpose | Autonomous LLM pretraining experimentation (5-min time-budget BPE training) |
| Package manager | `uv` (Python) |
| Upstream | karpathy/autoresearch |

## Dependency Analysis

### Python Dependencies (pyproject.toml)

| Package | Source | Version | Status |
|---------|--------|---------|--------|
| kernels | PyPI | >=0.11.7 | SAFE — HuggingFace's kernels package |
| matplotlib | PyPI | >=3.10.8 | SAFE — standard plotting |
| numpy | PyPI | >=2.2.6 | SAFE — standard |
| pandas | PyPI | >=2.3.3 | SAFE — standard |
| pyarrow | PyPI | >=21.0.0 | SAFE — standard parquet I/O |
| requests | PyPI | >=2.32.0 | SAFE — standard HTTP |
| rustbpe | PyPI | >=0.1.0 | SAFE — Karpathy's BPE crate |
| tiktoken | PyPI | >=0.11.0 | SAFE — OpenAI's official tokenizer |
| torch | https://download.pytorch.org/whl/cu128 | ==2.9.1 | SAFE — official PyTorch CUDA 12.8 wheel index |

The `[tool.uv.sources]` block pins `torch` to the official PyTorch wheel index (`pytorch-cu128`), which is a legitimate practice for installing CUDA-specific PyTorch builds.

### Node Dependencies (package.json)

`package.json` is a minimal stub (no dependencies, no install scripts). It exists only for project metadata and is not used at runtime.

## Code Analysis

### Checks Performed

| Category | Status |
|----------|--------|
| Command Execution (`os.system`, `subprocess`, `eval`, `exec`) | None detected |
| Network Calls (data exfiltration) | Only legitimate HuggingFace dataset downloads |
| File Operations (suspicious writes) | Only `~/.cache/autoresearch/` writes — expected |
| Obfuscated Code (base64, hex eval) | None detected |
| Hidden Files | `.env` (empty), `.gitignore`, `.python-version`, `.beads/.gitignore` — all benign |
| Hardcoded Credentials | None detected |
| Post-install Scripts | None — pure `uv sync` install |

### Findings

**Network operations:** `prepare.py` downloads parquet shards from `https://huggingface.co/datasets/karpathy/climbmix-400b-shuffle/resolve/main` via `requests.get` with retries and timeout — a standard, expected dataset fetch.

**Kernel loading:** `train.py` uses HuggingFace's `kernels` library to load Flash Attention 3 from `varunneal/flash-attention-3` (Hopper) or `kernels-community/flash-attn3` (other) at runtime. These are public HF repos and an established pattern for loading optimized CUDA kernels.

**File writes:** Restricted to `~/.cache/autoresearch/` (data and tokenizer cache). No writes to system directories or user dotfiles.

**No `eval` / `exec` / `subprocess` / `os.system`** anywhere in the source.

**No environment variable exfiltration:** `os.environ` is set only for two harmless PyTorch/HF Hub knobs (`PYTORCH_ALLOC_CONF`, `HF_HUB_DISABLE_PROGRESS_BARS`).

## Security Concerns

None.

## Recommendations

1. Run `uv sync` to install dependencies.
2. Run `uv run prepare.py` (one-time) to download data + train tokenizer.
3. Run `uv run train.py` to verify the training pipeline works end-to-end.
4. Requires a single NVIDIA GPU with sufficient VRAM (tested on H100; ~45 GB peak per the README example).

## Conclusion

This is a fork of a well-known, reputable open-source LLM training repository by Andrej Karpathy. All dependencies are official, all network calls are legitimate dataset/kernel fetches, and there are no suspicious code patterns. Verdict: **CLEAN — safe to install and run.**
