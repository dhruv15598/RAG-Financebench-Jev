# Optional local decision runtimes

These adapters are the implementations exercised on the development machine. They are optional: the default Jev path needs neither their weights nor their dependencies. Run them inside Linux/WSL in separate environments. A clean-machine installation of these optional runtimes has not been verified.

## Download

Install `huggingface_hub` in your chosen environment, then run from the repository root:

```sh
python local_deciders/download.py coreai2
python local_deciders/download.py decider4-q4
```

Choose only the models you need. Downloads are pinned to revisions and saved under ignored `.runtime/deciders`. They are large. The downloader does not install CUDA, build inference engines, or start a server.

## CoreAI 2B

The tested environment uses PyTorch 2.8 with CUDA 12.8, Transformers 5.17, FastAPI and Uvicorn. Use the appropriate PyTorch installation for your hardware. CoreAI uses its separately pinned inference source revision (`15ab28e`) to apply `temperature_by_type` correctly.

Example from the repository root, with the runtime Python environment activated:

```sh
DECIDER_MODEL="$PWD/.runtime/deciders/coreai2" \
DECIDER_INFER_ROOT="$PWD/.runtime/deciders/coreai-source" \
DECIDER_DEVICE=cuda DECIDER_ALLOW_CUDA=1 \
DECIDER_MODEL_ALIAS=decider-2b-coreai DECIDER_PORT=8011 \
python local_deciders/decider_server.py
```

CPU operation uses `DECIDER_DEVICE=cpu`; its speed and system-RAM requirements need separate validation. The dashboard configuration's `device` field must agree with the launch command.

## Decider Q4

Use a separate environment with `llama-cpp-python==0.3.35`, NumPy, FastAPI and Uvicorn. Our CUDA runtime required a source build with a compatible CUDA toolkit; installing the default CPU wheel does not enable GPU inference. Follow llama-cpp-python's installation instructions for your platform.

```sh
python local_deciders/q4_server.py \
  --model .runtime/deciders/decider4-q4/decider-4b.v2-Q4_K_M.gguf \
  --native-source .runtime/deciders/decider4-source \
  --port 8012 --ctx 32768 --gpu-layers -1
```

The native source directory is required and is downloaded alongside the GGUF; it supplies the checkpoint prompt and calibration configuration. For CPU execution, use `--gpu-layers 0` and set the dashboard entry to `device: cpu`.

The adapter uses the native plain prompt, direct option logits and temperature 1.935. It rejects oversized input rather than silently cutting evidence. It does not use a generic chat template.

## Connect to the dashboard

Put the corresponding command, paths, port and alias in a private `DECISION_MODELS_CONFIG` file based on the repository example. Use `/usr/bin/env` followed by `NAME=value` arguments for environment variables; commands are argument lists, not shell strings. Enable the entry only when configured. Do not manually start the server at the same time: the dashboard manages its lifecycle and refuses an occupied port.

See `HARDWARE.md` for memory policy. Passing integration tests means the request completed; it does not establish that a checker correctly identifies numerical errors.
