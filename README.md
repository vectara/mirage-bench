<!--- BADGES: START --->
[![HF Datasets](https://img.shields.io/badge/%F0%9F%A4%97-datasets-yellow)](https://huggingface.co/collections/nthakur/mirage-bench-naacl25-67ddb6166a7938a37436a455)
[![GitHub - License](https://img.shields.io/github/license/vectara/mirage-bench?logo=github&style=flat&color=green)][#github-license]
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/mirage-bench?logo=pypi&style=flat&color=blue)][#pypi-package]
[![PyPI - Package Version](https://img.shields.io/pypi/v/mirage-bench?logo=pypi&style=flat&color=orange)][#pypi-package]

[#github-license]: https://github.com/vectara/mirage-bench/blob/master/LICENSE
[#pypi-package]: https://pypi.org/project/mirage-bench/
<!--- BADGES: END --->

# MIRAGE-BENCH: Benchmarking LLM Generation Across Multiple Languages

This repository provides an easy way to perform the following four objectives:

1. Generate RAG-based answers to mulitilingual questions with support to many open-sourced LLMs tied with vLLM and closed-source LLMs via APIs such as Azure OpenAI, Cohere, Anthropic etc.
2. Evaluate multilingual RAG answers based on multiple heuristic features, such as support, fluency, or even automatic ones using open-sourced LLMs supported in vLLM.
3. LLM-as-a-Judge design to compare pairwise multilingual RAG answers, and train a bradley-terry model (with bootstrapping) to construct an offline multilingual RAG arena.
4. Train a surrogate judge (linear regression) to train and bootstrap the expensive LLM-as-a-judge design learnt using heuristic features.

For more information, checkout out our publication:
- [MIRAGE-Bench: Automatic Multilingual Benchmark Arena for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2410.13716) (Accepted at NAACL 2025 Main Conference)

## Installation

We recommend **Python 3.9+** and installing the latest version of **[vLLM](https://docs.vllm.ai/en/latest/getting_started/installation.html)**.

**Install with pip**

```
pip install -U mirage-bench
```

**Install from sources**

Alternatively, you can also clone the latest version from the [repository](https://github.com/vectara/mirage-bench) and install it directly from the source code:

````
pip install -e .
```` 

## Getting Started

Make sure you have the latest **[vLLM](https://docs.vllm.ai/en/latest/getting_started/installation.html)** repository installed correctly.

### 1. Multilingual RAG Answer Generation

First download a pretrained embedding a.k.a. Sentence Transformer model.

````python
# export AZURE_OPENAI_ENDPOINT="xxxxx"
# export AZURE_OPENAI_API_KEY="xxxx"

from mirage_bench import util
from mirage_bench.generate import AzureOpenAIClient

client = AzureOpenAIClient(model_name_or_path="gpt-4o-mini")

### Prompts_dict contains query_id as key and prompt as value
prompts_dict = util.load_prompts(
    dataset_name="nthakur/mirage-bench", 
    language_code="en", # or "ar", "bn" ... 18 languages supported
    split="dev" # only dev split is available in mirage-bench
) 
query_ids = list(prompts_dict.keys())
outputs = client.batch_call(
    prompts=list(prompts_dict.values()),
    temperature=0.1,
    max_new_tokens=2048,
)
#### output contains the List of RAG outputs
````
Similarly, you can even generate answers with a single/multiple GPU inference with [vLLM generation](https://github.com/vectara/mirage-bench/blob/main/examples/generation/vllm_generation.py).

### 2. Heuristic \& Automatic RAG Evaluation

After generating RAG answers, we evaluate the quality of the response using heuristic features:

```python
from mirage_bench import util
from mirage_bench.evaluate import RougeBleuEvaluator

evaluator = RougeBleuEvaluator(language_code="en")

# Load the documents (relevant & non-relevant)
documents = util.load_documents(
    dataset_name="nthakur/mirage-bench", 
    language_code="en", 
    split="dev"
)

# Load the multilingual RAG predictions available for 20+ models.
# In this example, we are evaluating: meta-llama/Meta-Llama-3-8B-Instruct
predictions = util.load_predictions(
    dataset_name="nthakur/mirage-bench-output",
    language_code="en",
    split="dev",
    model_name="meta-llama/Meta-Llama-3-8B-Instruct",
)

# Need to load the reference model, i.e., gold predictions
# This step is not necessary in all heuristic features
reference_predictions = util.load_predictions(
    dataset_name="nthakur/mirage-bench-output",
    language_code="en",
    split="dev",
    model_name="gpt-4-azure",
)

# Evaluate the predictions
scores = evaluator.evaluate(
    predictions=predictions, 
    reference_predictions=reference_predictions, 
    documents=documents
)
# => query_id: {"answer_bleu": 0.9, "answer_rougeL": 0.75}
```

### 3. LLM-as-a-Judge Pairwise Evaluation

After generating RAG answers, we can also use a LLM as a judge to compare two RAG outputs and decide which output is better.

```python
from mirage_bench import util
from mirage_bench.evaluate import PairwiseLLMJudgeEvaluator

evaluator = PairwiseLLMJudgeEvaluator(
    client="azure_openai",
    model_name_or_path="gpt-4o-mini"
)

# Load the documents (relevant & non-relevant)
documents = util.load_documents(
    dataset_name="nthakur/mirage-bench", 
    language_code="en", 
    split="dev"
)

# Load the queries available in the split
queries = util.load_queries(
    dataset_name="nthakur/mirage-bench", 
    language_code="en", 
    split="dev"
)

# In this example we will evaluate two models:
# 1. meta-llama/Meta-Llama-3-8B-Instruct
# 2. meta-llama/Meta-Llama-3-70B-Instruct
models = [
    "meta-llama/Meta-Llama-3-8B-Instruct",
    "meta-llama/Meta-Llama-3-70B-Instruct"
]

for model_name in models:
    predictions[model_name] = util.load_predictions(
        dataset_name="nthakur/mirage-bench-output",
        language_code="en",
        split="dev",
        model_name=model_name,
    )

scores = evaluator.evaluate(
    predictions=predictions,
    all_model_names=models, # provide all model names
    documents=documents,
    queries=queries
)
# IMP: model_A and model_B are randomly switched
# => [{"query_id": 1, 
#      "judge": "gpt-4o-mini", 
#      "model_A": "meta-llama/Meta-Llama-3-8B-Instruct", 
#      "model_B": "meta-llama/Meta-Llama-3-70B-Instruct", 
#      "output": <judge_output>,
#      "verdict": A/B/Tie.
#    }]
```

## Application Examples

You can use this framework for:

- [Multilingual RAG Generation](https://github.com/vectara/mirage-bench/tree/main/examples/generation)
- [Heuristic RAG Evaluations](https://github.com/vectara/mirage-bench/tree/main/examples/heuristic_evals)
- [Arena RAG Evaluations](https://github.com/vectara/mirage-bench/tree/main/examples/arena_evals)
- [Surrogate Judge Training \& Inference](https://github.com/vectara/mirage-bench/tree/main/examples/surrogate_judge)

## Citing & Authors

This work was done in a collaboration between Vectara and University of Waterloo.

If you find this repository helpful, feel free to cite our publication [MIRAGE-Bench: Automatic Multilingual Benchmark Arena for Retrieval-Augmented Generation Systems](https://arxiv.org/abs/2410.13716):

```bibtex 
@article{thakur-mirage-bench:2024,
  author       = {Nandan Thakur and
                  Suleman Kazi and
                  Ge Luo and
                  Jimmy Lin and
                  Amin Ahmad},
  title        = {MIRAGE-Bench: Automatic Multilingual Benchmark Arena for Retrieval-Augmented
                  Generation Systems},
  journal      = {CoRR},
  volume       = {abs/2410.13716},
  year         = {2024},
  url          = {https://doi.org/10.48550/arXiv.2410.13716},
  doi          = {10.48550/ARXIV.2410.13716},
  eprinttype    = {arXiv},
  eprint       = {2410.13716},
  timestamp    = {Wed, 27 Nov 2024 09:01:16 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2410-13716.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```

Maintainer: [Nandan Thakur](https://github.com/thakur-nandan), PhD Student @ University of Waterloo

Don't hesitate to open an issue if something is broken (and it shouldn't be) or if you have further questions.

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
