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

DATASET = "alpha/dataset/data_nonvalid_after_change_500.json"

with open("./llama/prompt_file.txt", "r") as f:
    messages = list(json.loads(f.read()))

def generate_function(original_function, function_name, argument_name, change_to):
	prompt = f'''Given a python function \'{function_name}\', we want to replace the parameter \'{argument_name}\' with \'{change_to}\', with semantics and logics of the function preserved. 
    Here is the function\n{original_function}'''
	messages.extend({"role": "user", "content":prompt})
	completion = client.chat.completions.create(
	    model="codellama/CodeLlama-7b-Instruct-hf", 
		messages=messages, 
		max_tokens=500
	)
	changed_function = completion.choices[0].message.content
	print(changed_function)
	return changed_function


def evaluate():
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
