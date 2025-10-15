from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnableSequence
# REQUIRES mistral (or other llm) running locally on a ollama server, https://ollama.com/download
# TODO: include output in system prompt, genereate multiple candidate prompts with .batch()

from metrics import bert_scoring, compression_score

minimizer_llm = ChatOllama(model="mistral", temperature=0, num_predict=200)
tester_llm = ChatOllama(model="mistral", temperature=0, num_predict=200)

# System prompt for minimization, subject to change
minimizer_prompt = ChatPromptTemplate.from_messages([
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

# Minimize the prompt
def minimize(inputs: dict):
    print("Current prompt: ", inputs["prompt"])
    msg = minimizer_prompt.format_messages(prompt=inputs["prompt"]) # add output when system prompt works
    response = minimizer_llm.invoke(msg)
    
    return {**inputs, "new_prompt": response.content}

# Test the minimized prompt
def test_minimized_prompt(inputs: dict):
    print("Proposed prompt:", inputs["new_prompt"])
    output = tester_llm.invoke(inputs["new_prompt"])

    return {**inputs, "new_output": output.content}

# Evaluate new prompt and output.
def evaluate(inputs: dict, bert_w=0.5, comp_w=0.5):
    comp_score = compression_score(inputs["new_prompt"], inputs["original_prompt"])
    bert_score = bert_scoring([inputs["new_output"]], [inputs["original_output"]])[0].item()
    total_score = (1-bert_score) * bert_w + comp_score * comp_w

    return {**inputs, "score": total_score, "bert_score": bert_score, "compression_score": comp_score}

def iterative_minimization(inputs: dict, steps=3):
    inputs["prompt"] = inputs["original_prompt"]
    inputs["output"] = inputs["original_output"]
    history = []

    for i in range(steps):
        print(f"\n--- Step {i+1} ---")

        result = full_chain.invoke(inputs)
        history.append(result)

        print("Comption Score:", result["compression_score"])
        print("BERT Score:", result["bert_score"])
        print("Score:", result["score"])

        inputs["prompt"] = result["new_prompt"]
        #inputs["output"] = result["new_output"]

    return history


minimize_chain = RunnableLambda(minimize)
test_chain = RunnableLambda(test_minimized_prompt)
evaluator_chain = RunnableLambda(evaluate)

full_chain = minimize_chain | test_chain | evaluator_chain

# Test input
test = {
    "original_prompt": "Explain the process of photosynthesis in simple terms for a 10th grade science class.",
    "original_output": "Photosynthesis is the process plants use to convert sunlight into energy. They take in carbon dioxide and water, and with the help of sunlight, they produce glucose and oxygen."
}

history = iterative_minimization(test, steps=3)

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
