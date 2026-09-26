# Choosing models for your machine

The dashboard defaults to Qwen 3.5 2B when installed. The saved 27B experiment was run on a 16 GB GPU; it is not the minimum requirement for using this repository.

| Hardware | Starting configuration | Status |
| --- | --- | --- |
| 8 GB GPU | Qwen 2B + remote Jev | Recommended starting point; not tested on an actual 8 GB card |
| 8 GB GPU, fully local | Qwen 2B + Decider Q4, loaded in separate phases | Candidate configuration; validate peak memory on your machine |
| Limited GPU memory, enough system RAM | Qwen on GPU + a CPU-capable decision runtime | CPU adapter configuration required; latency will differ |
| 16 GB GPU | Small Qwen + small local checker | Concurrent residency is being evaluated; default remains sequential |
| Any GPU | Large Qwen + large checker | Use sequential loading; each model still needs to fit independently or support CPU offload |

## RAM and VRAM are different

System RAM is the computer's main memory. VRAM belongs to the GPU. Keeping a checker in system RAM can avoid GPU contention, but requires CPU inference or an explicit offload implementation. Keeping its files in the operating system's cache is not the same as keeping a ready model running.

Jev runs remotely and takes no local model VRAM. Local weights, inference buffers, context and caches all consume memory. Download size alone is not a memory budget. Other applications and the desktop also use VRAM.

## Current loading policy

Local checks run sequentially: release connected Ollama models, load the checker, check evidence, stop the checker, generate with Qwen, release Qwen, reload the checker and check the answer. Cleanup releases both on completion or failure. This trades reload time for lower peak memory. The UI reports loading separately from evaluation.

The dashboard does **not** yet promise to keep local checkers permanently resident. A resident mode must account for both models' peak memory, context, model switching and cleanup before it can be enabled safely. No silent evidence truncation should be used to make a combination fit.

On the test machine, a CoreAI 2B + Qwen 2B residency probe completed correctly and reduced the second Qwen request from 30.09 seconds including loading to 1.12 seconds warm. Total device usage reached 9975 MiB; that includes the desktop and is not a model-only allocation measurement or an 8 GB fit guarantee. Both models were unloaded afterward.

For Decider Q4 at 32768-token runtime context, llama.cpp reported approximately 4773 MiB across model, KV, recurrent and compute CUDA buffers. This is substantially more than its 2.71 GB download. It excludes some driver/runtime overhead and the generator's memory. Concurrent residency on 8 GB remains unverified; sequential loading is the starting configuration.

## Configuration

`decision-models.example.json` contains per-checker `min_free_mib` thresholds. They are conservative startup guards, not measured peak allocations or guarantees. Do not lower them just to bypass an error. GPU telemetry uses the first NVIDIA GPU; multi-GPU placement is not managed here. Without telemetry, free memory cannot be verified.

For a CPU-capable adapter, set `device` to `cpu` in its entry **and configure the launch command to actually run on CPU**. The device field only controls the dashboard's GPU preflight; it does not move weights. Leave `device` as `cuda` for GPU runtimes. CPU execution still requires sufficient system RAM. Adapter support must be tested before claiming it works.

`QWEN_MIN_FREE_MIB` controls the generation-phase free-memory guard (default 6000 MiB). Ollama controls its own loading/offload behavior. An oversized model may run partly on CPU and be much slower; sequential loading cannot make its individual GPU requirements smaller.

The dashboard keeps the experiment's 8192-token context and 384-token output budget. Changing context to save memory requires checking that the complete evidence and answer budget still fit. Do not compare truncated evidence against full-evidence results as if they were the same experiment.

## Before calling a configuration supported

Run a complete evidence check → generation → answer check with a long evidence case, then repeat it warm. Record total/free VRAM, system RAM, exact models and quantizations, context, load time, inference time and whether CPU offload occurred. Test switching models and a failed/cancelled run. Our 16 GB measurements do not establish an 8 GB guarantee.
