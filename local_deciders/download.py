"""Download an optional pinned decision checkpoint into the ignored runtime folder."""
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download, hf_hub_download

MAP = '49564ddcfccafb6db563eb757c1d41e6c78dcb56'
CORE_SOURCE = '15ab28e8d8e4ec49728165b3426d81de5b77f864'
CORE = '284dfd3ea3eec2fa5902c09af10d0ea0647d6cbc'
GGUF = '2796fac5cdad018ef6d9a4a004735ff819f424a2'

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('model', choices=['coreai2', 'decider4-q4'])
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1] / '.runtime/deciders')
    a = p.parse_args()
    if a.model == 'coreai2':
        snapshot_download('mlboydaisuke/decider-2b-coreai-ft', revision=CORE, local_dir=a.root/'coreai2')
        snapshot_download('Mapika/decider-4b', revision=CORE_SOURCE, local_dir=a.root/'coreai-source',
                          allow_patterns=['decider/*', 'decider_config.json', 'LICENSE*'])
    else:
        hf_hub_download('mindchain/decider-4b-v2-GGUF', revision=GGUF,
                        filename='decider-4b.v2-Q4_K_M.gguf', local_dir=a.root/'decider4-q4')
        snapshot_download('Mapika/decider-4b', revision=MAP, local_dir=a.root/'decider4-source',
                          allow_patterns=['decider/*', 'decider_config.json', 'LICENSE*'])
    print('Checkpoint saved under', a.root.resolve())

if __name__ == '__main__':
    main()
