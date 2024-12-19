import requests
import json
import alpha.evaluate_metrics as alpha
from transformers import AutoTokenizer
import transformers
import torch
from huggingface_hub import InferenceClient

model = "meta-llama/CodeLlama-7b-Instruct-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device="cuda"
        )


torch.cuda.empty_cache()


with open("./llama/prompt_file.txt", "r") as f:
    messages = json.loads(f.read())['data']

def generate_prompt():
    few_shot_prompt = ""
    for ele in messages:
        few_shot_prompt += f'''#{ele["role"]}: {ele["content"]}'''
def generate_function(original_function, function_name, argument_name, change_to):
    prompt = f'''\n#User: Given a python function \'{function_name}\', we want to replace the parameter \'{argument_name}\' with \'{change_to}\', with semantics and logics of the function preserved. 
    Here is the function\n{original_function}'''
    torch.cuda.empty_cache()
    res = pipeline(few_shot_prompt+prompt,do_sample=True,top_k=10,temperature=0.1,top_p=0.95,num_return_sequences=1,eos_token_id=tokenizer.eos_token_id, max_length=len(original_function)+100) 
    print('\nres', res[0])
    changed_function = res[0]['generated_text']

    print('changed', changed_function)
    return changed_function


def evaluate(DATASET = "./alpha/dataset/data_alpha_non_valid_after_change_500.json"):
    f = open(DATASET, 'r')
    data_res = json.loads(f.read())

    total_accuracy = 0.0
    total_count = 0
    for data in data_res:
        total_count += 1
        argument_name = data["target_argument"]
        change_to = data["change_to"]
        changed_function = data["changed_function"]
        original_function = data["original_function"]
        function_name = data["function_name"]
        inputs = data["inputs"]

        changed_function = generate_function(original_function, function_name, argument_name, change_to)
        print("original", original_function)
        print("changed", changed_function)
        accuracy = alpha.evaluate(original_function, changed_function, argument_name, inputs)
        total_accuracy += accuracy

    print("final accuracy: ", total_accuracy/total_count)
