"""
export OMP_NUM_THREADS=6
export OUTPUT_PATH=/u3/n3thakur/projects/vectara/vectara-translation/datasets
export HF_HOME=/u3/n3thakur/projects/cache
export DATASETS_HF_HOME=/u3/n3thakur/projects/cache
export AZURE_OPENAI_ENDPOINT="xxxxx"
export AZURE_OPENAI_API_KEY="xxxx"

for lang in en; do
for model in gpt-3.5-turbo-azure
do
    python examples/generation/openai_inference.py --language $lang --split dev \
    --model $model \
    --cache_dir /u3/n3thakur/projects/cache \
    --dataset_name "nthakur/mirage-eval" \
    --output_dir $OUTPUT_PATH/$model/mirage-eval-test/ \
    --filename $lang-mirage-eval-raft-eval-dev  --temperature 0.1 \
    --batch_size 12  --num_gpus 1 --concurrency 1 --filter_start 0
done done
"""

import argparse

from mirage_bench.generate import HFDataset

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", default=None)
    parser.add_argument("--temperature", required=False, type=float, default=0.3)
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--dataset_name", default=None)
    parser.add_argument("--split", default="train")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--filename", default=False)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--max_tokens", type=int, default=2048)
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--filter_start", type=int, default=0)
    parser.add_argument("--filter_end", type=int, default=None)

    args = parser.parse_args()

    hf_dataloader = HFDataset(args.dataset_name, args.language, args.split, args.cache_dir)
    hf_dataloader.load_dataset()

    print(hf_dataloader.hf_dataset)
