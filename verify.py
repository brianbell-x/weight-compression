"""Entry point for the lossless verifier, so the public command stays short.

    uv run verify.py <org>/<repo>       # stream a model from Hugging Face
    uv run verify.py --model <path>     # verify a local .safetensors file/dir

Streaming (an HF repo id) runs stream_validate.py: one shard on disk at a time,
bit-exact check on every BF16 tensor, decoded byte-split measurement. A local --model
path runs reproduce.py: full encode + decode round-trip, SHA-256 checked.
"""

import runpy
import sys

TOOLS = "src/tools"
script = (
    f"{TOOLS}/reproduce.py" if "--model" in sys.argv else f"{TOOLS}/stream_validate.py"
)
sys.argv[0] = script
runpy.run_path(script, run_name="__main__")
