# Verification performed

On 24 September 2026 in an existing Linux/WSL research environment:

- Six local tests passed: generation data boundary, incomplete-output handling, embedding/cache validation, snapshot/manifest consistency, no adapter text cropping, and actual local Tesseract OCR of an image-only PDF.
- An isolated live smoke run processed the pinned PepsiCo PDF using upstream Docket extraction/chunking, creating 35 chunks. Live EmbeddingGemma embeddings, default Docket hybrid retrieval, Qwen3 reranking, Qwen3.5 2B generation and HTML rendering completed. The fresh answer stated the correct one-percentage-point increase and finished with `stop` in about 28 seconds. This is one integration example, not a measured accuracy score.
- The packaged reranker adapter loaded the existing official 0.6B weights on CPU. A relevant EPS passage scored 0.99965 versus 0.000033 for a football passage; wrong-model requests returned HTTP 400. No truncation was applied.
- CLI help and rendering the genuine saved ten-case snapshot succeeded.

Reproduce offline checks with `python -m pytest tests -q`. With local services ready, `python tests/live_smoke.py` runs the one-report integration test in ignored cache/output folders. The normal `run.py --prepare` uses every report in the manifest; the smoke deliberately uses only one.

A fresh Windows 11 installation and a new complete 28-report ingestion were not executed for this packaging test. OCR availability, installation permissions, first model downloads, GPU memory and service startup vary by machine. No new Jev gateway requests were made during packaging; its optional client reuses the previously exercised request/validation format. Historical snapshot verdicts are not independent correctness labels.


Qwen 3.8 addition: all ten fixed cases completed on Qwen 3.8 27B UD-IQ2_S with fresh Jev answer checks, using saved baseline passages and prechecks. Median generation time was 3.57 seconds after warm-up. Nine responses were supported, including two abstentions; this is not an accuracy score. The exact model installer passed SHA-256 validation and real Ollama registration against the existing downloaded file. Python compilation and Windows setup parsing passed. The new HTML was checked for all ten answer/reference pairs. Clean-machine download/bootstrap and simultaneous reranker/generator memory use remain unverified.
