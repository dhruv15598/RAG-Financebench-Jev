# Data and attribution

FinanceBench public dataset: Patronus AI, https://huggingface.co/datasets/PatronusAI/financebench . Dataset revision and exact source hashes are in data/manifest.json. Dataset licensed CC BY-NC 4.0 (https://creativecommons.org/licenses/by-nc/4.0/): attribution and noncommercial use required. The issuer reports retain their original rights. This package is a private, noncommercial research demonstration.

The compact historical snapshot includes ten questions, reference answers and retrieved report excerpts for inspection. It is derived from FinanceBench and issuer filings; it is not the full corpus. The historical outputs are generated answers and automated Jev judgments, not independently verified financial advice or accuracy labels. Reference answers never enter generation or retrieval.

Retrieval/ingestion dependency: Docket by Aditya Menon, https://github.com/adityam23/docket , GPL-3.0-only, pinned commit d97a18063e37894d02aafc3b35da0c516bf14c83. It is installed from upstream, not copied into this repository. This package adds orchestration, local serving, reporting and optional Jev checks; it does not replace Docket's retriever. Refer to each model's own license before redistribution; no model weights are included.
