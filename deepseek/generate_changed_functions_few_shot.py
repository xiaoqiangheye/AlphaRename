from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json
import alpha.evaluate_metrics as alpha

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()


prompt = '''You are a code programmer, we would like you to perform a alpha-renaming task a given function,
changing a function argument name to another name and preserve the semantics. \n
You probably need to rename other names in the functions to avoid name conflits. \n

Here is the function

``function begins``
'''

with open("./deepseek/prompt_file.txt", "r") as f:
    messages = json.loads(f.read())['messages']

with open("./starcoder/prompt_file_cot.txt", "r") as f:
    few_shot_cot = json.load(f)
    few_shot_cot_promt = []
    data = few_shot_cot[0]
    cot = data["step-by-step thoughts"]
    changed = data["changed_function"]
    original = data["original_function"]
    name = data["function_name"]
    argument = data["target_argument"]
    change_to = data["change_to"]
    few_shot_cot_promt = few_shot_cot_promt + [{"role": "user", "content": f'''Given a python function {name} we want to replace the parameter '{argument}' with '{change_to}', with semantics and logics of the function preserved. 
Here is the original function:
{original}
Analyze the problems step by step and return the replaced function. 
'''
}]
    few_shot_cot_promt = few_shot_cot_promt + [{"role":'assistant', "content": 
f'''
{cot}

```python
{changed}
```
'''
}]

def evaluate_cot(path = './alpha/dataset/data_alpha_non_valid2.json'):
    f = open(path, 'r')
    data_res = json.loads(f.read())
    torch.cuda.empty_cache()
    count, total_accu = 0, 0
    for data in data_res:
        count += 1
        argument_name = data["target_argument"]
        change_to = data["change_to"]
        changed_function = data["changed_function"]
        original_function = data["original_function"]
        function_name = data["function_name"]
        inputs = data["inputs"]

        input_text = {'role': 'user', 'content': f'''Given a python function {function_name} we want to replace the parameter '{argument_name}' with '{change_to}'. Here is the original function:
{original_function}
Analyze the problems step by step and return the replaced function, Mark the returned result with ```python'''}
        print(few_shot_cot_promt+[input_text])
        inputss = tokenizer.apply_chat_template(few_shot_cot_promt+[input_text], add_generation_prompt=True, return_tensors="pt").to(model.device)
        model_output = model.generate(inputss, max_length=len(original_function)+800, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        output = tokenizer.decode(model_output[0], skip_special_tokens=True)
        print(output)
        changed_function = output.split('### Response:')[-1].split('```python')[1].split("```")[0]
        print(changed_function)
        accuracy = alpha.evaluate(original_function, changed_function, function_name, inputs)
        total_accu += accuracy
        print(total_accu/count)
        data["changed_function"] = changed_function
    print(total_accu / len(data_res))
    outfile = open("./alpha/evaluation_data/deepseek_data_alpha_non_valid2_few_shot.json", 'w')
    outfile.write(json.dumps(data_res))
    outfile.close()

def evaluate(path = './alpha/dataset/data_alpha_non_valid2.json'):
    f = open(path, 'r')
    data_res = json.loads(f.read())
    torch.cuda.empty_cache()
    count, total_accu = 0, 0
    for data in data_res:
        count += 1
        argument_name = data["target_argument"]
        change_to = data["change_to"]
        changed_function = data["changed_function"]
        original_function = data["original_function"]
        function_name = data["function_name"]
        inputs = data["inputs"]

        input_text = {'role': 'user', 'content': f'''Given a python function {function_name} we want to replace the parameter '{argument_name}' with '{change_to}'. Here is the original function:\n\n{original_function}\n\n'''}
        inputss = tokenizer.apply_chat_template(messages+[input_text], add_generation_prompt=True, return_tensors="pt").to(model.device)
        model_output = model.generate(inputss, max_length=len(original_function)+800, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
        changed_function = tokenizer.decode(model_output[0], skip_special_tokens=True).split('### Response:')[-1]
        print(changed_function)
        accuracy = alpha.evaluate(original_function, changed_function, function_name, inputs)
        print(accuracy)
        total_accu += accuracy
        data["changed_function"] = changed_function
    print(total_accu / len(data))
    outfile = open("../alpha/evaluation_data/deepseek_data_alpha_non_valid2_few_shot.json", 'w')
    outfile.write(json.dumps(data_res))
    outfile.close()

