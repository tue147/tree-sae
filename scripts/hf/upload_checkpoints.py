"""Upload converted SAEs (``convert_checkpoints.py`` output) to the public Hugging Face repository.

HF_TOKEN=... python scripts/hf/upload_checkpoints.py --folder hf_release
"""

from __future__ import annotations

import argparse
import os

from huggingface_hub import HfApi

from tree_sae.models.io import HF_REPO_ID


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--folder", default="hf_release")
    parser.add_argument("--repo_id", default=HF_REPO_ID)
    parser.add_argument("--private", action="store_true", help="Create the repository as private")
    args = parser.parse_args()

    api = HfApi(token=os.environ.get("HF_TOKEN"))
    api.create_repo(args.repo_id, repo_type="model", private=args.private, exist_ok=True)
    api.upload_folder(repo_id=args.repo_id, folder_path=args.folder, commit_message="Release Tree SAE checkpoints")
    print(f"Uploaded {args.folder} to https://huggingface.co/{args.repo_id}")


if __name__ == "__main__":
    main()
