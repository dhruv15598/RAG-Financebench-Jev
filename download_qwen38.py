"""Download and register the exact Qwen 3.8 quantization used in the saved run."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

REPO = 'unsloth/Qwen3.8-27B-GGUF'
REVISION = '4ca720788d1e01f1bff70c033e0d0028fd02e502'
FILE = 'Qwen3.8-27B-UD-IQ2_S.gguf'
SHA256 = '7897d2c5a5cee46aef50895141b2c8a0803c1185f3d03c4fda4cd137a7ad77fe'
ALIAS = 'qwen3.8:27b-iq2s'

def verify(path):
    with path.open('rb') as stream:
        actual = hashlib.file_digest(stream, 'sha256').hexdigest()
    if actual != SHA256:
        raise ValueError('Model checksum mismatch; file was not registered.')

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model-dir', type=Path,
                        default=Path(__file__).resolve().parent / '.runtime/models/qwen38')
    parser.add_argument('--ollama-url', default='http://127.0.0.1:11435')
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    print(json.dumps({'repo': REPO, 'revision': REVISION, 'file': FILE,
                      'sha256': SHA256, 'alias': ALIAS, 'ollama_url': args.ollama_url}))
    if args.check_only:
        return
    path = args.model_dir.resolve() / FILE
    if not path.is_file():
        from huggingface_hub import hf_hub_download
        path = Path(hf_hub_download(REPO, FILE, revision=REVISION,
                                   local_dir=str(args.model_dir.resolve())))
    verify(path)
    modelfile = path.parent / 'Modelfile'
    modelfile.write_text(f'FROM "{path.as_posix()}"\nPARAMETER num_ctx 8192\n', encoding='utf-8')
    subprocess.run(['ollama', 'create', ALIAS, '-f', str(modelfile)], check=True,
                   env={**os.environ, 'OLLAMA_HOST': args.ollama_url})
    print(f'Ready: python run.py --model {ALIAS} --limit 10 --jev')

if __name__ == '__main__':
    main()
