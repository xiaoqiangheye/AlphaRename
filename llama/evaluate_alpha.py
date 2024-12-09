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


def generate_function(original_function, function_name, argument_name, change_to):
    prompt = f'''Given a python function {function_name}
we want to replace the parameter {argument_name} with {change_to} and with semantics and logics preserved. 
Mark the start and the end of function with ¥¥¥.
Here is the function:
<{original_function}>.
Here is the replaced function, no explanation needed:
¥¥¥
def {function_name}({change_to}):
'''
    res = pipeline(prompt,do_sample=True,top_k=10,temperature=0.1,top_p=0.95,num_return_sequences=1,eos_token_id=tokenizer.eos_token_id, max_length=len(original_function)+100)
#    print(res[0])
    changed_function = res[0]['generated_text'].split('¥¥¥')[2]
 #   print(changed_function)
    return changed_function


def generate_cot_function(original_function, function_name, argument_name, change_to):
    prompt = f'''{few_shot_cot_promt}
Human: Given a python function {function_name} we want to replace the parameter '{argument_name}' with '{change_to}', with semantics and logics of the function preserved. 
Here is the function
{original_function}

Analyze the problems step by step and return the replaced function
Assistant:
'''

    res = pipeline(prompt,do_sample=True,top_k=10,temperature=0.1,top_p=0.95,num_return_sequences=1,eos_token_id=tokenizer.eos_token_id, max_length=200)
    print(res[0])
    changed_function = res[0]['generated_text']
    print(changed_function)
    return changed_function



def evaluate_cot(path='./alpha/dataset/data_alpha_non_valid2.json'):
    f = open(path, 'r')
    data_res = json.loads(f.read())
    total_count = 0
    total_accuracy = 0
    for data in data_res:
        total_count += 1
        argument_name = data["target_argument"]
        change_to = data["change_to"]
        # changed_function = data["changed_function"]
        original_function = data["original_function"]
        function_name = data["function_name"]
        inputs = data["inputs"]

        try:
            changed_function = generate_cot_function(
                    original_function, function_name, argument_name, change_to)
            data["changed_function"] = changed_function
        except:
            continue

        accuracy = alpha.evaluate(
                original_function, changed_function, function_name, inputs)
        print(accuracy)
        total_accuracy += accuracy
    
    outfile = open("./alpha/evaluation_data/starcoder_data_alpha_non_valid2_cot.json", 'w')
    outfile.write(json.dumps(data_res))
    outfile.close()
    print("final accuracy:", total_accuracy/total_count)
    return total_accuracy, total_count


def evaluate(path = "alpha/dataset/data_alpha_347_valid.json"):
	f = open(path, 'r')
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
        data["changed_function"] = changed_function
        print("original", original_function)
        print("changed", changed_function)
    
        outfile = open("./alpha/evaluation_data/llama_data_alpha_347_valid.json", 'w')
        outfile.write(json.dumps(data_res))
        outfile.close()
#        accuracy = alpha.evaluate(original_function, changed_function, argument_name, inputs)
 #       total_accuracy += accuracy

  #  print("final accuracy: ", total_accuracy/total_count)
