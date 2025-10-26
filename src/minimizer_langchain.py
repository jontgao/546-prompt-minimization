from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda
# REQUIRES mistral (or other llm) running locally on a ollama server, https://ollama.com/download
# TODO: include output in system prompt, genereate multiple candidate prompts with .batch()
# TODO: make compatible with main, flag for vllm or ollama

from metrics import BERTScoreScorer, CompressionLengthScorer


class PromptMinimizerLangChain:
    def __init__(self, config):
        self.config = config
        self.bert_scorer  = BERTScoreScorer()
        self.compression_scorer = CompressionLengthScorer()
        
        if getattr(self.config, "ollama", False):
            from langchain_ollama import ChatOllama
            self.llm = ChatOllama(model="mistral", temperature=self.config.temperature, num_predict=self.config.max_token_length)
        else:
            from langchain_community.llms import VLLM
            self.llm = VLLM(model=self.config.model, temperature=self.config.temperature, max_new_tokens=self.config.max_token_length)

        # system prompt for minimization, subject to change
        self.minimizer_prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a prompt engineer. Your job is to rewrite and minimize prompts to be as short and concise as possible "
                "The user will provide you with an original prompt."
                "Your job is to generate a new, shorter prompt that will produce the same output"
                "ONLY adapt the original prompt to be shorter while preserving its meaning."
                "DO NOT use the original output in your response"
            )),
            ("user", (
                "Original prompt: \n\n{prompt}\n\n "
                #"Original prompt: \n\n{output}\n\n " # Spent too much time trying to get this to work, if I include it the LLM copies it... 
                "Short prompt:"
            )),
        ])

        # Define the chains
        self.minimize_chain = RunnableLambda(self.minimize)
        self.test_chain = RunnableLambda(self.test_minimized_prompt)
        self.evaluator_chain = RunnableLambda(self.evaluate)
        self.full_chain = self.minimize_chain | self.test_chain | self.evaluator_chain

    def __call__(self, original_prompt: str, original_output: str):
        inputs = {
            "original_prompt": original_prompt,
            "original_output": original_output
        }
        return self.iterative_minimization(inputs, steps=self.config.num_iterations)

    # Minimize the prompt
    def minimize(self, inputs: dict):
        print("Current prompt: ", inputs["prompt"])

        msg = self.minimizer_prompt.format_messages(prompt=inputs["prompt"]) # add output when system prompt works
        response = self.llm.invoke(msg)
        content = response.content if hasattr(response, "content") else response
        return {**inputs, "new_prompt": content}

    # Test the minimized prompt
    def test_minimized_prompt(self, inputs: dict):
        print("Proposed prompt:", inputs["new_prompt"])

        output = self.llm.invoke(inputs["new_prompt"])
        content = output.content if hasattr(output, "content") else output
        return {**inputs, "new_output": content}

    # Evaluate new prompt and output.
    def evaluate(self, inputs: dict):
        comp_score = self.compression_scorer.compute_score(inputs["new_prompt"], inputs["original_prompt"])
        bert_score = self.bert_scorer.compute_score([inputs["new_output"]], [inputs["original_output"]])[0].item()
        total_score = (1-bert_score) * self.config.bert_score_weight + comp_score * self.config.compression_weight

        return {**inputs, "score": total_score, "bert_score": bert_score, "compression_score": comp_score}

    def iterative_minimization(self, inputs: dict, steps=3):
        inputs["prompt"] = inputs["original_prompt"]
        inputs["output"] = inputs["original_output"]
        history = []

        for i in range(steps):
            print(f"\n--- Step {i+1} ---")

            result = self.full_chain.invoke(inputs)
            history.append(result)

            print("Compression Score:", result["compression_score"])
            print("BERT Score:", result["bert_score"])
            print("Score:", result["score"])

            inputs["prompt"] = result["new_prompt"]
            #inputs["output"] = result["new_output"]

        return history



if __name__ == "__main__":
    class Config:
        #ollama = True  
        model = 'TinyLlama/TinyLlama-1.1B-Chat-v1.0'
        temperature = 0.0
        bert_score_weight = 0.5
        compression_weight = 0.5
        num_iterations = 3
        max_token_length = 200
    
    # Test input
    original_prompt = "Explain the process of photosynthesis in simple terms for a 10th grade science class."
    original_output = "Photosynthesis is the process plants use to convert sunlight into energy. They take in carbon dioxide and water, and with the help of sunlight, they produce glucose and oxygen."

    #history = iterative_minimization(test, steps=3)
    temp = PromptMinimizerLangChain(Config())
    history = temp(original_prompt, original_output)

    """
    # Run the full chain once
    result = full_chain.invoke(test)

    # Print output
    print("Prompt Minimization Result:\n")
    print("Original Prompt:", result["original_prompt"])
    print("Shortened Prompt:", result["new_promp"])
    print("New Output:", result["new_output"])
    print("BERT Score:", result["bert_score"])
    print("Compression Score:", result["compression_score"])
    print("Score:", result["score"])
    """