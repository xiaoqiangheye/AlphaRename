import requests
import json
import alpha.evaluate_metrics as alpha
from transformers import AutoTokenizer
import transformers
import torch

model = "meta-llama/CodeLlama-7b-hf"

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    torch_dtype=torch.float16,
	device="cuda"
)

# API_URL = "https://api-inference.huggingface.co/models/meta-llama/CodeLlama-7b-hf"
# headers = {"Authorization": "Bearer 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"}
DATASET = "alpha/dataset/data_nonvalid_after_change_500.json"

def generate_function(original_function, function_name, argument_name, change_to):
	prompt = f'''Given a python function {function_name}
we want to replace the parameter {argument_name} with {change_to} and with semantics and logics preserved. 
Mark the start and the end of function with ¥¥¥.
Here is the function:
<{original_function}>.
Here is the replaced function, no explanation needed:
¥¥¥
def {function_name}({change_to}
'''
	res = pipeline(prompt,do_sample=True,top_k=10,temperature=0.1,top_p=0.95,num_return_sequences=1,eos_token_id=tokenizer.eos_token_id, max_length=200)
	print(res[0])
	changed_function = res[0]['generated_text'].split('¥¥¥')[2]
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