# Verification performed

On 24 September 2026 in an existing Linux/WSL research environment:

- Six local tests passed: generation data boundary, incomplete-output handling, embedding/cache validation, snapshot/manifest consistency, no adapter text cropping, and actual local Tesseract OCR of an image-only PDF.
- An isolated live smoke run processed the pinned PepsiCo PDF using upstream Docket extraction/chunking, creating 35 chunks. Live EmbeddingGemma embeddings, default Docket hybrid retrieval, Qwen3 reranking, Qwen3.5 2B generation and HTML rendering completed. The fresh answer stated the correct one-percentage-point increase and finished with `stop` in about 28 seconds. This is one integration example, not a measured accuracy score.
- The packaged reranker adapter loaded the existing official 0.6B weights on CPU. A relevant EPS passage scored 0.99965 versus 0.000033 for a football passage; wrong-model requests returned HTTP 400. No truncation was applied.
- CLI help and rendering the genuine saved ten-case snapshot succeeded.

Reproduce offline checks with `python -m pytest tests -q`. With local services ready, `python tests/live_smoke.py` runs the one-report integration test in ignored cache/output folders. The normal `run.py --prepare` uses every report in the manifest; the smoke deliberately uses only one.

Setup and launch commands are maintained in [README.md](README.md).

A fresh Windows 11 installation and a new complete 28-report ingestion were not executed for this packaging test. OCR availability, installation permissions, first model downloads, GPU memory and service startup vary by machine. No new Jev gateway requests were made during the initial packaging checks; its optional client reuses the previously exercised request/validation format. Historical snapshot verdicts are not independent correctness labels.

Known clean-machine gaps: the Windows bootstrap has not been run end to end on a clean laptop; WSL installation, Ubuntu first-run setup, sudo permissions, network access, Ollama installation, first model downloads and GPU memory remain machine-dependent. The Linux helper intentionally does not install OS packages. The batch runner requires the prepared index, Ollama and the reranker. The dashboard reuses bundled passages and requires only Ollama plus gateway access; it does not require the index or reranker. Jev additionally requires an explicitly supplied `AI_GATEWAY_API_KEY`; no key is stored by the setup scripts.


Qwen 3.8 addition: all ten fixed cases completed on Qwen 3.8 27B UD-IQ2_S with fresh Jev answer checks, using saved baseline passages and prechecks. Median generation time was 3.57 seconds after warm-up. Nine responses were supported, including two abstentions; this is not an accuracy score. The exact model installer passed SHA-256 validation and real Ollama registration against the existing downloaded file. Python compilation and Windows setup parsing passed. The new HTML was checked for all ten answer/reference pairs. Clean-machine download/bootstrap and simultaneous reranker/generator memory use remain unverified.

## Live dashboard verification (24 September 2026)

Browser runs used saved passages, fresh Jev evidence checks, streamed Qwen 3.8 answers and fresh Jev answer checks. PepsiCo completed with 1 percentage point (supported); Jev round trips were 0.38 s and 0.61 s, generation 39.84 s including loading. Verizon also completed (supported); Jev took 0.83 s and 0.89 s, generation 67.47 s including loading. Reference answers now appear on question selection, including when a live run fails. They are not model inputs.

`python -m pytest tests -q`: nine tests passed, including three dashboard tests covering cross-site/unknown-input rejection, upstream-error redaction, run-lock release and bounded transient retries. An earlier gateway failure was not reproduced in four direct checks; the dashboard now exposes safe error details and retries temporary failures once. This does not establish production reliability. The dashboard reuses saved retrieval; it is not a new retrieval benchmark. Browser verification used Windows with services in WSL. The new launcher has not been tested on a clean machine.
