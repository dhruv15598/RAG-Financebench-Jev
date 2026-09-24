# Finance RAG with Jev

Built on [Docket by Aditya](https://github.com/adityam23/docket). Docket supplies document ingestion and hybrid retrieval. This project adds **Jev checks through the Vercel AI Gateway**, with scripts, a notebook and reports for evaluating financial-document answers.

The flow is: retrieve passages → rerank → expand to source pages → Jev checks evidence sufficiency → Qwen answers → Jev checks answer support. Jev is optional. Its scores are model judgments, not proven probabilities of correctness. Diagnostic runs retain cases even when the evidence check rejects them.

## Why FinanceBench?

We use [FinanceBench, published by Patronus AI](https://huggingface.co/datasets/PatronusAI/financebench), to test question answering over company financial reports. Its public dataset contains 150 questions with reference answers and supporting evidence. Questions cover finding reported figures, comparing periods and calculating financial metrics, making it useful for examining retrieval failures and numerical mistakes.

This repository runs **ten fixed development questions** against a **28-report corpus** from our earlier pilot. Docket retrieves passages from the reports, Qwen generates an answer, and optional Jev checks judge evidence sufficiency and answer support. The report then displays the FinanceBench reference answer for comparison. Reference answers and annotated evidence are not supplied as answer hints to Qwen or Jev.

These ten cases are a selected subset, not a full FinanceBench evaluation or an unbiased accuracy estimate. Jev's verdicts are separate from the benchmark references: an answer can be judged supported and still be wrong. FinanceBench uses a noncommercial license; see [data attribution and licensing](DATA_LICENSE.md).

## View saved results

The latest saved run is [Qwen 3.8 27B UD-IQ2_S](data/qwen38-results.html), with [raw results](data/qwen38-results.json). Download the HTML and open it in a browser; GitHub displays HTML source. All ten cases completed, with median answer generation of 3.57 seconds after warm-up. Jev supported nine responses, including two abstentions; this is not 90% answer accuracy. The run reused the same retrieved passages and evidence checks as the 2B baseline, then generated new answers and new Jev answer checks.

Open `data/snapshot.html` in a browser. No installation or key is needed. The ten historical cases include retrieved passages, model answers, Jev judgments and FinanceBench references. They are selected development cases, not a random test set.

`Demo.ipynb` is the editable notebook version. Open it in VS Code or Jupyter; executing cells requires Python with the repository dependencies.

## Set up on Windows 11

1. Clone this repository in GitHub Desktop, or download and extract its ZIP.
2. Open PowerShell in the repository folder and run:

   ```powershell
   .\setup.ps1
   ```

   Setup uses the selected Ubuntu distribution in WSL, installs the Python runtime and Tesseract, installs Ollama inside that same distribution, downloads the embedding, answer and reranker models, then downloads and indexes the 28 source reports. The first run needs internet access and several GB of disk space. If Windows requests administrator approval or a restart for WSL, complete that and Ubuntu's first-run account setup, then rerun the command. Ubuntu may ask for its password to install packages. Check resolved paths without changing the machine with `.\setup.ps1 -CheckOnly`.

3. Generate a fresh report:

   ```powershell
   .\run.ps1 -Limit 10
   ```

4. Open `results.html` in the new timestamped folder under `outputs/`.

Defaults: `qwen3.5:2b`, `embeddinggemma:latest`, and `Qwen/Qwen3-Reranker-0.6B`. Ollama listens on `127.0.0.1:11435`; the local reranker listens on `127.0.0.1:11436`. An NVIDIA GPU is recommended; CPU execution is slower. All services run in the same Ubuntu WSL instance. After restarting the computer, rerun `setup.ps1` to reuse or start services and verify the prepared cache. A complete installation on a clean Windows laptop has not yet been verified; see `TESTING.md` for completed checks.

## Linux

The Python pipeline is shared across platforms. PowerShell and WSL are only the Windows convenience route.

Install Python 3.12, Git, Tesseract with English language data, and [Ollama](https://ollama.com/download) first. On Debian/Ubuntu, install Git and OCR with `sudo apt install git tesseract-ocr tesseract-ocr-eng`. Other Linux distributions should use their own package manager.

From the repository folder:

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python setup_local.py
python run.py --limit 10
```

`setup_local.py` checks prerequisites, downloads models, starts services and prepares the index. It does not install system packages or create another environment; use an active Python 3.12 environment with Git, Tesseract and Ollama already on `PATH`. Rerun it after restarting the machine. `python setup_local.py --check-only` checks the Python version, files and prerequisites without starting services or downloading anything. Linux is the supported non Windows path.

For NVIDIA acceleration on Linux, use the matching PyTorch build from the [official installation selector](https://pytorch.org/get-started/locally/) in the same environment. The reranker uses CUDA when available and otherwise CPU. Ollama handles answer and embedding model acceleration separately.


## Models and switching

| Role | Model | Use in this repository |
| --- | --- | --- |
| Answer generation (default) | Qwen 3.5 2B | Local baseline; saved report uses Q8_0. Setup pulls the Ollama tag, which can change upstream. |
| Answer generation (optional) | Qwen 3.8 27B UD-IQ2_S | Exact revision and checksum pinned; latest saved ten-case report. |
| Evidence and answer checks | TypeSafe Jev | Optional Vercel AI Gateway calls; enabled with `-Jev` / `--jev`. |
| Embeddings | EmbeddingGemma 300M | Turns questions and passages into vectors for retrieval. |
| Reranking | Qwen3-Reranker-0.6B | Ranks retrieved passages before answer generation. |

After installing the desired answer model, switch with the model argument. This keeps the same corpus and retrieval configuration; the embedding index does not need rebuilding.

```powershell
# Windows
.\run.ps1 -Model qwen3.5:2b -Limit 10 -Jev
.\run.ps1 -Model qwen3.8:27b-iq2s -Limit 10 -Jev
```

```bash
# Linux
python run.py --model qwen3.5:2b --limit 10 --jev
python run.py --model qwen3.8:27b-iq2s --limit 10 --jev
```

Omit the Jev flag to run without gateway checks. Each command writes a separate timestamped report. Answer generation uses an 8,192-token context, temperature 0, thinking disabled and a 384-token output limit. Other Ollama models can be selected by their installed name but are not validated by this project; thinking/template support and memory requirements can differ. Qwen 2.5 1.5B was used in earlier research but is not included in this packaged comparison. Qwen 3.5 9B Q6_K was downloaded during model selection, but has no completed result set here.

## Install optional Qwen 3.8

The default remains Qwen 3.5 2B. To install the exact **Qwen 3.8 27B UD-IQ2_S** used in the latest report:

```powershell
# Windows, in the repository folder
.\setup.ps1 -Qwen38
.\run.ps1 -Model qwen3.8:27b-iq2s -Limit 10 -Jev
```

```bash
# Linux, with the Python environment activated
python setup_local.py --qwen38
python run.py --model qwen3.8:27b-iq2s --limit 10 --jev
```

Configure the gateway key as described below before running with Jev, or omit `-Jev` / `--jev`. If the runtime is already ready, Linux users can run `python download_qwen38.py` alone to download/register the model. The installer pins the Hugging Face revision and verifies the model's SHA-256 before registration. Weights stay in ignored `.runtime/`; they are not committed.

The file is 8.37 GB. On our RTX 5070 Ti 16 GB, Ollama reported about 8.3 GiB of GPU memory for this model at an 8,192-token context. The timed saved run had the reranker unloaded and reused saved evidence; it does not establish memory or speed for the full concurrent pipeline. Fresh runs retrieve and rerank again. Other hardware can require CPU offload or smaller settings. A clean-machine installation of this option has not been tested. `download_qwen38.py` uses the package `.runtime/models/qwen38` location; changing `-RuntimeDir` does not relocate that optional GGUF unless you pass its own `--model-dir`.

## Enable Jev

Create a Vercel AI Gateway API key and set a spending limit. Jev receives the question and retrieved passages; its answer check also receives Qwen's answer. Only use documents permitted to be sent to that service.

Enter the key without displaying it, then run:

```powershell
$gatewaySecret = Read-Host 'Vercel AI Gateway key' -AsSecureString
$env:AI_GATEWAY_API_KEY = [System.Net.NetworkCredential]::new('', $gatewaySecret).Password
.\run.ps1 -Limit 10 -Jev
Remove-Item Env:AI_GATEWAY_API_KEY
```

The key is passed to WSL through the environment. Do not put it in source files, notebooks or commits. Without `-Jev`, no Jev requests are made.

On Linux, this hidden prompt runs the same experiment without putting the key in shell history:

```bash
python -c "import getpass,os,subprocess,sys; os.environ['AI_GATEWAY_API_KEY']=getpass.getpass('Gateway key: '); sys.exit(subprocess.call([sys.executable,'run.py','--limit','10','--jev']))"
```

## Options

```powershell
.\run.ps1 -Limit 1
.\run.ps1 -Limit 10 -Jev -Out outputs/my-run
.\run.ps1 -Prepare
```

Initial setup already prepares the index. `-Prepare` prepares or resumes a compatible index. `-Out` must name a folder that does not exist; interrupted runs leave partial JSON for inspection and should be rerun with a new output folder. Retrieval failures are recorded per question so remaining questions can continue. `-Model` selects a model already installed in the same Ollama service; it does not download one. `-Distro` selects a different installed WSL distribution and `-RuntimeDir` selects the runtime/venv location (the retrieval cache remains in the repository’s `cache/` folder) when the defaults are unsuitable.

## Live dashboard

After setup, run `.\dashboard.ps1` on Windows, or `python dashboard.py` in the configured Linux environment. Open http://localhost:7860. Enter your Vercel key at the hidden terminal prompt, or supply `AI_GATEWAY_API_KEY` in the server environment. The key stays on the server. Ollama must be running on port 11435; Linux users can override this with `OLLAMA_URL`. `dashboard.ps1` accepts `-Distro`, `-RuntimeDir` and `-Port`.

A timeout can occur after the gateway has processed a request. Retrying it may therefore incur a second charge; the dashboard retries at most once per check.

Choose one of the ten saved FinanceBench cases and an installed Qwen model. The dashboard reuses the experiment's selected source pages, expanded to include their complete indexed text and makes **new** Jev evidence checks, streams a **new** Qwen answer, then asks Jev to check that answer. It does not repeat retrieval or indexing and does not require the reranker or a prepared index once its Python dependencies and answer model are available. Both models receive the same expanded page text. Historical reports retain their original chunks and verdicts. The benchmark reference is shown below the model responses for the selected question, even if a live call fails. It is never sent to either model.

Generation uses an 8,192-token context, temperature 0 and up to 384 output tokens. Incomplete generations are not sent for approval. Timings include model loading and network overhead. Jev scores are model judgments, not calibrated accuracy estimates. The dashboard is local-only, runs one demonstration at a time, and does not save new runs. A successful run makes two Jev requests. A temporary network failure or HTTP 408/429/5xx can trigger one visible retry per check, which can add requests and latency. Authentication and response-validation failures are not retried. Safe failure details are saved locally in `outputs/dashboard-errors.jsonl`; no keys or upstream response bodies are recorded. Stop with Ctrl+C.

## Files and credit

| File | Purpose |
| --- | --- |
| `setup.ps1` / `run.ps1` | Windows setup and execution |
| `setup_local.py` | Linux service and model setup |
| `run.py` | Calls Docket retrieval, Qwen and optional Jev |
| `jev.py` | Jev client using Vercel AI Gateway |
| `reranker.py` | Local endpoint for the Qwen reranker |
| `render.py` / `Demo.ipynb` | Report and editable notebook |
| `data/` | Source manifest and saved research results |
| `dashboard.py` / `dashboard.html` | Live model streaming and Jev checks in a local browser |

Docket is installed from a pinned upstream commit. Its embeddings, keyword search and hybrid retrieval are not new methods introduced here. The local reranker adapter and experiment/reporting code connect those components for this experiment. FinanceBench reference answers are evaluation-only and are never sent to Qwen or Jev. Dataset use is subject to its noncommercial license; see `DATA_LICENSE.md`.

## Complete-page evidence fix

`evidence.py` expands selected chunks to all indexed chunks on the same document/page, removes overlapping text and deduplicates repeated page hits. It preserves retrieval rank and does not use reference answers or company-specific routing. This restores table continuations such as AMD’s investing and financing totals without rebuilding embeddings. It does not reconstruct table cells or join tables across different pages, and it does not prevent unrelated companies from being retrieved.

Fresh batch runs apply this automatically. The dashboard uses `data/dashboard-evidence.json`, a separate evidence-only artifact; historical answer reports are unchanged. Rebuild it from a prepared corpus with:

```bash
python evidence.py --index cache/index/corpus.jsonl
```

Expansion refuses evidence exceeding 3,600 words rather than cutting off page text. This is a conservative word budget, not an exact model-token guarantee; unusually token-dense pages can still need a larger context or a smaller retrieval set.
