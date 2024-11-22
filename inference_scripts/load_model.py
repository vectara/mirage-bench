from transformers import AutoModel, AutoTokenizer
import argparse

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")

    args = parser.parse_args()

    print(f"Loading model: {args.model}")

    model = AutoModel.from_pretrained(args.model)
    tokenizer = AutoTokenizer.from_pretrained(args.model)