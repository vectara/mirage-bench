from typing import List

class ChainOfThoughtPrompt:
    def __init__(self):
        self.prompt = """Please act as an impartial judge and evaluate the quality of the responses provided by two
                        AI assistants to the user question displayed below. Your evaluation should consider
                        correctness and helpfulness based on the reference context. You will be given assistant A's answer, 
                        and assistant B's answer. Your job is to evaluate which assistant's answer is better. You should 
                        independently provide an answer with a reasoning to the user question using the information provided 
                        in contexts written in {language}.  Cite parts of your reasoning based on the context within brackets 
                        [] as in the IEEE format. Please use the format of: ##Reason: <reason> ##Answer: <answer>.
                        Then compare both assistants' answers with your answer. Identify and correct any mistakes. 
                        Avoid any position biases and ensure that the order in which the responses were presented 
                        does not influence your decision. Do not allow the length of the responses to influence your evaluation. 
                        Do not favor certain names of the assistants. Be as objective as possible. After providing your 
                        explanation, output your final verdict by strictly following this format: "[[A]]" if
                        assistant A is better, "[[B]]" if assistant B is better, and "[[C]]" for a tie.
                        
                        [User Question]
                        {question}

                        [Reference Context]
                        {context}

                        [The Start of Assistant A's Answer]
                        {answer_a}
                        [The End of Assistant A's Answer]
                        
                        [The Start of Assistant B's Answer]
                        {answer_b}
                        [The End of Assistant B's Answer]"""


    def __call__(self, query: str, contexts: List[str], answers: List[str]) -> str:

        context_str = "\n\n".join(contexts)

        return self.prompt.format(context=context_str, question=query, answer_A=answers[0], answer_B=answers[1])