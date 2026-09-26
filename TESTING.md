# Verification performed

On 24 September 2026 in an existing Linux/WSL research environment:

- Six local tests passed: generation data boundary, incomplete-output handling, embedding/cache validation, snapshot/manifest consistency, no adapter text cropping, and actual local Tesseract OCR of an image-only PDF.
- An isolated live smoke run processed the pinned PepsiCo PDF using upstream Docket extraction/chunking, creating 35 chunks. Live EmbeddingGemma embeddings, default Docket hybrid retrieval, Qwen3 reranking, Qwen3.5 2B generation and HTML rendering completed. The fresh answer stated the correct one-percentage-point increase and finished with `stop` in about 28 seconds. This is one integration example, not a measured accuracy score.
- The packaged reranker adapter loaded the existing official 0.6B weights on CPU. A relevant EPS passage scored 0.99965 versus 0.000033 for a football passage; wrong-model requests returned HTTP 400. No truncation was applied.
- CLI help and rendering the genuine saved ten-case snapshot succeeded.

Reproduce offline checks with `python -m pytest tests -q`. With local services ready, `python tests/live_smoke.py` runs the one-report integration test in ignored cache/output folders. The normal `run.py --prepare` uses every report in the manifest; the smoke deliberately uses only one.

Setup and launch commands are maintained in [README.md](README.md).

A fresh Windows 11 installation and a new complete 28-report ingestion were not executed for this packaging test. OCR availability, installation permissions, first model downloads, GPU memory and service startup vary by machine. No new Jev gateway requests were made during the initial packaging checks; its optional client reuses the previously exercised request/validation format. Historical snapshot verdicts are not independent correctness labels.

Known clean-machine gaps: the Windows bootstrap has not been run end to end on a clean laptop; WSL installation, Ubuntu first-run setup, sudo permissions, network access, Ollama installation, first model downloads and GPU memory remain machine-dependent. The Linux helper intentionally does not install OS packages. The batch runner requires the prepared index, Ollama and the reranker. The dashboard reuses bundled passages and requires Ollama plus a configured decision backend; it does not require the index or reranker. Jev additionally requires an explicitly supplied `AI_GATEWAY_API_KEY`; no key is stored by the setup scripts.


Qwen 3.8 addition: all ten fixed cases completed on Qwen 3.8 27B UD-IQ2_S with fresh Jev answer checks, using saved baseline passages and prechecks. Median generation time was 3.57 seconds after warm-up. Nine responses were supported, including two abstentions; this is not an accuracy score. The exact model installer passed SHA-256 validation and real Ollama registration against the existing downloaded file. Python compilation and Windows setup parsing passed. The new HTML was checked for all ten answer/reference pairs. Clean-machine download/bootstrap and simultaneous reranker/generator memory use remain unverified.

## Live dashboard verification (24 September 2026)

Browser runs used saved passages, fresh Jev evidence checks, streamed Qwen 3.8 answers and fresh Jev answer checks. PepsiCo completed with 1 percentage point (supported); Jev round trips were 0.38 s and 0.61 s, generation 39.84 s including loading. Verizon also completed (supported); Jev took 0.83 s and 0.89 s, generation 67.47 s including loading. Reference answers now appear on question selection, including when a live run fails. They are not model inputs.

`python -m pytest tests -q`: nine tests passed, including three dashboard tests covering cross-site/unknown-input rejection, upstream-error redaction, run-lock release and bounded transient retries. An earlier gateway failure was not reproduced in four direct checks; the dashboard now exposes safe error details and retries temporary failures once. This does not establish production reliability. The dashboard reuses saved retrieval; it is not a new retrieval benchmark. Browser verification used Windows with services in WSL. The new launcher has not been tested on a clean machine.

## Table-continuation regression

Eleven tests pass after page expansion. Regression checks confirm the corrected AMD page contains operating 3,565, investing 1,999 and financing (3,264), with the year and units. All ten cases preserve their originally selected document/page order. Additional tests cover overlap removal, numeric chunk order, document boundaries, duplicate pages and oversized/missing-page failures. Corrected evidence was rebuilt from the existing full corpus. New model-answer quality is not established by these deterministic tests; historical verdicts were not reused for the expanded evidence.

The user confirmed that the live dashboard worked after the corrected-evidence restart. This is a manual smoke-check confirmation, not a new benchmark score or an independently captured answer evaluation.

## Historical decision model integration (26 September 2026)

Twenty-seven tests passed in the existing WSL Python environment after adding mocked local lifecycle coverage. The selectable registry contains Jev, Kev 0.8B, Kev 4B, Decider 4B v2 and Decider 2B CoreAI. The same success, initialization-failure and evaluation/generation-failure lifecycle cases run for Kev 0.8B and Decider 2B CoreAI. Tests verify local selection without a gateway key, Qwen-only generation, exact evidence/answer phases, GPU release before checker loading, checker termination before Qwen, failure cleanup, occupied-port refusal, allow-listed configuration and invalid probability rejection. No test loads weights, consumes gateway credits or stops another service. Browser script syntax is checked separately. Actual checkpoint runs and GPU-fit verification remain environment-dependent.


A live browser run on 26 September used Qwen 3.8 27B UD-IQ2_S and Kev 0.8B on the same full-page PepsiCo evidence. Both checks and streamed generation completed. Checker round trips were 0.42 s and 0.36 s, initialization 14.89 s and 7.23 s, and generation including loading 42.97 s. Qwen returned 1 percentage point. Kev selected supported at only 35.9%; this is a workflow check, not evidence of reliable accuracy. After cleanup Ollama reported no loaded models and total GPU usage returned to approximately 2.9 GiB.


## Dashboard pre-commit checks (26 September 2026)

35 automated tests passed after adding the Q4 registry entry and explicit CPU/GPU preflight configuration. Live dashboard API runs completed evidence check, fresh Qwen 3.5 2B streaming, answer check and final cleanup for Jev, Kev 0.8B, Kev 4B, Decider 4B v2 and Decider 2B CoreAI. These are one-question PepsiCo integration checks, not ten-question accuracy results. Decider Q4 also passed the same live dashboard flow after native-tokenizer parity validation. Qwen 2.5 1.5B and Qwen 3.5 9B passed separate dashboard runs with Kev 0.8B; Qwen 3.8 27B UD-IQ2_S also passed. All four installed answer models and all six decision backends have completed the dashboard flow. This is component coverage, not an exhaustive live UI run of all 24 combinations. The CoreAI runtime now uses the pinned inference revision that applies its per-type temperature settings. No commit has been made for these additions.

## Streamlined decision models

The active dashboard registry now contains Jev, Decider 2B CoreAI and Decider 4B v2 Q4. Kev and BF16 Decider 4B remain historical trial subjects only. Current mocked lifecycle tests exercise both retained local checkers and configuration migration: retired IDs warn and are ignored, while unknown IDs still fail. The downloader exposes only CoreAI 2B and Q4; no model weights are deleted by this change.

Verification: 33 tests passed in the existing AIConsultancy WSL environment. Network-free downloader tests verify pinned source inclusion for CoreAI/Q4 and rejection of the retired BF16 download before any network request. No GPU model was loaded.

Separate live validation after restarting the dashboard passed for both packaged adapters with Qwen 3.5 2B on the saved PepsiCo evidence: CoreAI 2B completed in 39.164 seconds and Decider Q4 in 56.718 seconds. Each run completed the evidence check, streamed fresh generation, answer check and cleanup. The browser selector and dashboard API expose exactly the three retained choices. These are integration checks on the existing machine, not accuracy measurements or clean-install verification.
