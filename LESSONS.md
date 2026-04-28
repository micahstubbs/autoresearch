# Lessons

Append-only debugging and tuning insights for the autoresearch project.

---

## 2026-04-28T11:00 - peak_vram_mb is not what nvidia-smi shows

**Problem**: When tuning `DEVICE_BATCH_SIZE` for a 24 GB RTX 4090, the most natural first instinct is to read `nvidia-smi` during training to budget headroom. With BS=32, `nvidia-smi` reported 16 GB used, leaving ~8 GB free. That made BS=64 look infeasible (would double to 32 GB). But the script's own `peak_vram_mb` reading at end-of-run was only 11.7 GB.

**Root Cause**: `torch.cuda.max_memory_allocated()` (what `peak_vram_mb` reports) measures peak *active* allocations. `nvidia-smi memory.used` measures everything torch has *reserved* from the OS — including its caching allocator's free pool. The two diverge by gigabytes during training.

**Lesson**: For batch-size sizing, trust `torch.cuda.max_memory_allocated()`, not `nvidia-smi`. But trust `nvidia-smi` for the absolute ceiling — torch's reserved memory is what triggers an OOM, not its allocated peak.

**Prevention**: When a run completes, prefer the script-reported `peak_vram_mb` for headroom math. If the GPU is shared with a desktop/browser, subtract baseline `nvidia-smi memory.used` (before torch starts) from total VRAM to get the budget — not from torch's runtime peak.

---

## 2026-04-28T11:05 - OOM hits at torch.compile, not in the training loop

**Problem**: BS=64 OOMed almost immediately on a 4090. Expected the failure during the first forward pass at ~20 GB activations. Actual failure: `Tried to allocate 128.00 MiB. ... 19.65 GiB is allocated by PyTorch, and 57.37 MiB is reserved but unallocated` — before training started, during graph capture / compile.

**Root Cause**: `torch.compile(model, dynamic=False)` and `@torch.compile(fullgraph=True)` bake a static graph that allocates buffers (gradients, optimizer state, fused-kernel scratch, possibly compile workspaces) up-front. On a small consumer GPU with another process holding 3-4 GB, you can run out of memory *before* a single training step.

**Lesson**: A "smoke test" for batch size is whether torch.compile finishes — not whether the loss prints. If you see the OOM message during compile/init, the BS is too high regardless of how much activation memory you'd need at runtime.

**Prevention**: When pushing BS upward on a consumer GPU, watch the log for the first `step 00000` line; if it never appears and `nvidia-smi` shows >90% memory, you've hit a compile-time OOM, not a forward-pass OOM. Drop BS and retry — don't tune `expandable_segments`, `gc.collect`, or activation checkpointing first.

---

## 2026-04-28T11:10 - DEVICE_BATCH_SIZE choices are constrained to divisors of (TOTAL_BATCH_SIZE / MAX_SEQ_LEN)

**Problem**: Wanted to try BS=48 as a middle ground between BS=32 (works, 11.7 GB) and BS=64 (OOMs at compile). It's invalid: `train.py` asserts `TOTAL_BATCH_SIZE % (DEVICE_BATCH_SIZE * MAX_SEQ_LEN) == 0`.

**Root Cause**: With `TOTAL_BATCH_SIZE = 2**19 = 524288` and `MAX_SEQ_LEN = 2048`, `DEVICE_BATCH_SIZE` must divide `524288 / 2048 = 256`. Valid values: `1, 2, 4, 8, 16, 32, 64, 128, 256`. Nothing in between.

**Lesson**: Don't reach for a "halfway" batch size on this codebase without first changing `TOTAL_BATCH_SIZE`. The optimizer-step token count is a fixed hyperparameter — touching it changes the experiment, not just the memory.

**Prevention**: Before proposing a non-power-of-2 batch size, check `(TOTAL_BATCH_SIZE / MAX_SEQ_LEN)` and only suggest values that divide it.
