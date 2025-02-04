# from __future__ import annotations

# import os

# from openai import AzureOpenAI
# from tqdm.autonotebook import tqdm

# from .default import DefaultGenerator


# class OpenAIGenerator(DefaultGenerator):

#     def __init__(self, args):
#         super().__init__(args)
#         self.client = AzureOpenAI(
#             azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
#             api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#             api_version="2024-02-01",
#         )

#         self.deployment_name = None
#         if "gpt-3.5-turbo" in args.model:
#             self.deployment_name = "gpt-35-turbo"
#         elif "gpt-4o" in args.model:
#             self.deployment_name = "gpt-4o"
#         elif "gpt-4" in args.model:
#             self.deployment_name = "gpt-4"

#     def generate(self, prompt):
#         for data in tqdm(all_data, total=len(all_data), desc="Generating outputs..."):

#             prompt = data["prompt"]
#             query_id = data["query_id"]

#             try:
#                 response = client.chat.completions.create(
#                     model=self.deployment_name,
#                     messages=[{"role": "user", "content": f"{prompt}"}],
#                     temperature=args.temperature,
#                     n=1,
#                 )

#                 output = response.choices[0].message.content

#                 save_results(args.output_dir, final_output, output_filepath)


#             except Exception as e:
#                 print(f"Error for query_id: {query_id}. Error: {e}")





# if __name__ == '__main__':

#     parser = argparse.ArgumentParser()
#     parser.add_argument("--language", default=None)
#     parser.add_argument("--temperature", required=False, type=float, default=0.3)
#     parser.add_argument("--model", default="mistralai/Mistral-7B-Instruct-v0.2")
#     parser.add_argument("--cache_dir", default=None)
#     parser.add_argument("--dataset_name", default=None)
#     parser.add_argument("--split", default="train")
#     parser.add_argument("--output_dir", default=None)
#     parser.add_argument("--filename", default=False)
#     parser.add_argument("--batch_size", type=int, default=8)
#     parser.add_argument("--max_tokens", type=int, default=2048)
#     parser.add_argument("--num_gpus", type=int, default=1)
#     parser.add_argument("--concurrency", type=int, default=4)
#     parser.add_argument("--filter_start", type=int, default=0)
#     parser.add_argument("--filter_end", type=int, default=None)

#     args = parser.parse_args()

#     # Read one text file from S3. Ray Data supports reading multiple files
#     # from cloud storage (such as JSONL, Parquet, CSV, binary format).
#     # ds = ray.data.read_text("s3://anonymous@air-example-data/prompts.txt")
#     hf_dataset = datasets.load_dataset(args.dataset_name, args.language, split=args.split, cache_dir=args.cache_dir)
#     if args.filter_end is None: args.filter_end = len(hf_dataset)
#     # datasets_list = [datasets.Dataset.from_list(load_jsonl_file(i)) for i in .dataset_name]
#     # hf_dataset = datasets.concatenate_datasets(datasets_list)

#     os.makedirs(args.output_dir, exist_ok=True)

#     print(f"Loaded {len(hf_dataset)} prompts for {args.language}...")
#     output_filepath = f"{args.filename}-{args.filter_start}-{args.filter_end}.jsonl"

#     if os.path.exists(os.path.join(args.output_dir, output_filepath)):
#         print(f"File {output_filepath} already exists. No need to rerun the experiment.")
#         exit(0)

#     client = AzureOpenAI(
#         azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
#         api_key=os.getenv("AZURE_OPENAI_API_KEY"),
#         api_version="2024-02-01",
#     )

#     all_data = []

#     for idx, row in enumerate(hf_dataset):
#         prompt = row["prompt"]
#         query_id = row["query_id"]
#         positive_ids = row["positive_ids"]
#         negative_ids = row["negative_ids"]
#         all_data.append({
#             "prompt": prompt,
#             "query_id": query_id,
#             "positive_ids": positive_ids,
#             "negative_ids": negative_ids
#         })

#     all_data = all_data[args.filter_start:args.filter_end]
#     final_output = {}
#     idx = 0

#     for data in tqdm(all_data, total=len(all_data), desc="Generating outputs..."):
#         idx += 1

#         prompt = data["prompt"]
#         query_id = data["query_id"]
#         positive_ids = data["positive_ids"]
#         negative_ids = data["negative_ids"]

#         prompt_prefix = prompt.split("Instruction:")[0]
#         language = ISO_MAP[args.language]
#         prompt = prompt_prefix + f"Instruction: Provide an answer to the question using the information provided in contexts written in {language}. " + \
#         f"Additionally, provide a step-by-step explanation of your reasoning, demonstrating how you arrived at your answer in {language}. " + \
#         "Cite parts of your reasoning based on the context within brackets [] as in the IEEE format. " + \
#         "Please use the format of: ##Reason: {reason} ##Answer: {answer}."

#         try:
#             deployment_name = None
#             if "gpt-3.5-turbo" in args.model:
#                 deployment_name = "gpt-35-turbo"
#             elif "gpt-4o" in args.model:
#                 deployment_name = "gpt-4o"
#             elif "gpt-4" in args.model:
#                 deployment_name = "gpt-4"

#             response = client.chat.completions.create(
#                 model=deployment_name, # model = "deployment_name".
#                 messages=[{"role": "user", "content": f"{prompt}"}],
#                 temperature=args.temperature,
#                 n=1,
#             )

#             output = response.choices[0].message.content
#             final_output[idx] = {
#                 "outputs": {f"{args.model}": output},
#                 "prompt": prompt,
#                 "query_id": query_id,
#                 "positive_ids": positive_ids,
#                 "negative_ids": negative_ids
#             }

#             save_results(args.output_dir, final_output, output_filepath)


#         except Exception as e:
#             print(f"Error for query_id: {query_id}. Error: {e}")

