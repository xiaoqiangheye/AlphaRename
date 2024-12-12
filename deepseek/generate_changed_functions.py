from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()


with open("./starcoder/prompt_file_cot.txt", "r") as f:
    few_shot_cot = json.load(f)
    few_shot_cot_promt = ""
    for data in few_shot_cot:
        cot = data["step-by-step thoughts"]
        changed = data["changed_function"]
        original = data["original_function"]
        name = data["function_name"]
        argument = data["target_argument"]
        change_to = data["change_to"]
        few_shot_cot_promt += f'''Human: Given a python function {name} we want to replace the parameter '{argument}' with '{change_to}', with semantics and logics of the function preserved. 
Here is the function
{original}

Analyze the problems step by step and return the replaced function
Assistant:
{cot}

returned function:
{changed}
'''


def evaluate(path = './alpha/dataset/data_alpha_non_valid2.json'):
    f = open(path, 'r')
    data_res = json.loads(f.read())

    prompt = '''You are a code programmer, we would like you to perform a alpha-renaming task a given function,
    changing a function argument name to another name and preserve the semantics. \n
    You probably need to rename other names in the functions to avoid name conflits. \n

    Here is the function

    ``function begins``
    '''

    torch.cuda.empty_cache()
    for data in data_res:

        argument_name = data["target_argument"]
        change_to = data["change_to"]
        changed_function = data["changed_function"]
        original_function = data["original_function"]
        function_name = data["function_name"]
        inputs = data["inputs"]

        input_text = f'''Given a python function {function_name} we want to replace the parameter '{argument_name}' with '{change_to}'. Here is the original function:\n\n{original_function}\n\nHere is the replaced function, no explanation needed:\n\ndef {function_name}({change_to}):'''
        model_input = tokenizer(input_text, return_tensors="pt").to(model.device)
        model_output = model.generate(**model_input, max_length=len(original_function)+100)
        changed_function = tokenizer.decode(model_output[0], skip_special_tokens=True).split('Here is the replaced function, no explanation needed:')[1].split('A:')[0].strip()
        print(changed_function)
        data["changed_function"] = changed_function

    outfile = open("./alpha/evaluation_data/deepseek_data_alpha_non_valid2.json", 'w')
    outfile.write(json.dumps(data_res))
    outfile.close()


def evaluate_cot(path = './alpha/dataset/data_alpha_non_valid2.json'):
    f = open(path, 'r')
    data_res = json.loads(f.read())
    torch.cuda.empty_cache()
    for data in data_res:

        argument_name = data["target_argument"]
        change_to = data["change_to"]
        changed_function = data["changed_function"]
        original_function = data["original_function"]
        function_name = data["function_name"]
        inputs = data["inputs"]

        input_text = f'''{few_shot_cot_promt}
Human: Given a python function {function_name} we want to replace the parameter '{argument_name}' with '{change_to}', with semantics and logics of the function preserved. 
Here is the original function:\n\n{original_function}

Analyze the problems step by step and return the replaced function
Assistant:
'''
        model_input = tokenizer(input_text, return_tensors="pt").to(model.device)
        model_output = model.generate(**model_input, max_length=len(input_text)+100)
        changed_function = tokenizer.decode(model_output[0], skip_special_tokens=True).split('returned function:')[1].strip()
        print(changed_function)
        data["changed_function"] = changed_function

    outfile = open("./alpha/evaluation_data/deepseek_data_alpha_non_valid2.json", 'w')
    outfile.write(json.dumps(data_res))
    outfile.close()
