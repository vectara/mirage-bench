import datasets
from src.prompts.utils import load_prompt_template
from transformers import AutoTokenizer
from tqdm import tqdm
import time


LANGS = ["hi", "id", "it", "ja", "ko", "ms", "nl", "pt", "ru", "sw", "ta", "te", "th", "tr", "vi", "yo", "zh"]

def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]

def postprocess(text):
    text = text.replace("\n", " ")
    return " ".join(text.split())

prompt_cls = load_prompt_template("deita")
hf_dataset = datasets.load_dataset("mlabonne/orpo-dpo-mix-40k", split="train", cache_dir="/u3/n3thakur/projects/cache")

source_list = {lang: [] for lang in LANGS}
row_ids = {lang: [] for lang in LANGS}
turn_ids = {lang: [] for lang in LANGS}
translation_prompts = {lang: [] for lang in LANGS}
inputs_list = {lang: [] for lang in LANGS}
chosen_list = {lang: [] for lang in LANGS}
rejected_list = {lang: [] for lang in LANGS}
turns = {lang: [] for lang in LANGS}

max_turns, count = 6, 0

for idx, row in tqdm(enumerate(hf_dataset), total=len(hf_dataset)):
    chosen = row["chosen"]
    rejected = row["rejected"]
    source = row["source"]

    if len(chosen) <= max_turns and len(rejected) <= max_turns:
        turn_id = 0
        
        for chosen_chunk, rejected_chunk in zip(chunks(chosen, n=2), chunks(rejected, n=2)):
            if chosen_chunk[0]["role"] == "user":
                prompt = chosen_chunk[0]["content"]

            if chosen_chunk[1]["role"] == "assistant":
                chosen_string = chosen_chunk[1]["content"]
            
            if rejected_chunk[1]["role"] == "assistant":
                rejected_string = rejected_chunk[1]["content"]
            
            if chosen_string == rejected_string:
                for lang in LANGS:
                    positive_prompt = prompt_cls(question=prompt, chosen=chosen_string, language=lang)
                    translation_prompts[lang].extend([positive_prompt])
                    row_ids[lang].extend(["orpo_dpo_mix_" + str(idx)])
                    turns[lang].extend([turn_id])
                    turn_ids[lang].extend(["chosen"])
                    inputs_list[lang].extend([prompt])
                    chosen_list[lang].extend([chosen_string])
                    rejected_list[lang].extend([rejected_string])
                    source_list[lang].extend([source])

            else:
                for lang in LANGS:
                    positive_prompt = prompt_cls(question=prompt, chosen=chosen_string, language=lang)
                    negative_prompt = prompt_cls(question=prompt, chosen=rejected_string, language=lang)
                    
                    translation_prompts[lang].extend([positive_prompt, negative_prompt])
                    row_ids[lang].extend(["orpo_dpo_mix_" + str(idx), "orpo_dpo_mix_" + str(idx)])
                    turns[lang].extend([turn_id, turn_id])
                    turn_ids[lang].extend(["chosen", "rejected"])
                    inputs_list[lang].extend([prompt, prompt])
                    chosen_list[lang].extend([chosen_string, chosen_string])
                    rejected_list[lang].extend([rejected_string, rejected_string])
                    source_list[lang].extend([source, source])

            turn_id += 1

print("Length of translation prompts: ", len(translation_prompts[lang]))

for lang in LANGS:
    hf_dataset = datasets.Dataset.from_dict({
        "id": row_ids[lang],
        "turn": turns[lang],
        "en_prompt": inputs_list[lang], 
        "en_chosen": chosen_list[lang],
        "en_rejected": rejected_list[lang],
        "source": source_list[lang],
        "selection": turn_ids[lang], 
        "translation_prompt": translation_prompts[lang]
    })

    hf_dataset.push_to_hub("nthakur/orpo-dpo-mix-40k-flat", config_name=lang, private=False, split="train")
    time.sleep(60)
