from vllm import LLM, SamplingParams

# Answers to sample prompts.
# 1. उत्तर : 8 मिनिटे 20 सेकंद
# 2. Je kunt het gebruiken om; op te schrijven, op te tekenen, te vouwen, te knippen, origami te beoefenen, een papieren vliegtuig mee te maken, een collage mee te maken, te versnipperen.	
# 3. 答案：孔子在中国的鲁国（今山东省曲阜市）出生。	
# 4. James Buchanan est le seul président qui ne s'est jamais marié.


# Sample prompts.
prompts = [
    "[INST] answer the following question in Marathi: सूर्य किरण पृथ्वीवर पोहोचण्यास किती वेळ लागतो? [/INST]",
    "[INST] answer the following question in Dutch: Wat kun je doen met papier? [/INST]",
    "[INST] answer the following question in Chinese: 问题：孔子在哪里出生? [/INST]"
    "[INST] answer the following question in French: Quels président des États-Unis ne s'est jamais marié? [/INST]"
]
# Create a sampling params object.
sampling_params = SamplingParams(temperature=0.3, top_p=0.95, max_tokens=512)

# Create an LLM.
llm = LLM(model="google/gemma-7b-it", max_model_len=1024, max_num_seqs=1)
# Generate texts from the prompts. The output is a list of RequestOutput objects
# that contain the prompt, generated text, and other information.
outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")