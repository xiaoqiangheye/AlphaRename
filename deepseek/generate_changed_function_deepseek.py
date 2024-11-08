from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

path = '../../alpha/data_alpha.json'
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

    input_text = f'''Given a python function {function_name} we want to replace the parameter '{argument_name}' with '{change_to}'. Here is the original function:\n\n{original_function}\n\nHere is the replaced function, no explanation needed:\n\ndef {function_name}({change_to}):'''
    model_input = tokenizer(input_text, return_tensors="pt").to(model.device)
    model_output = model.generate(**model_input, max_length=len(original_function)+100)
    changed_function = tokenizer.decode(model_output[0], skip_special_tokens=True).split('Here is the replaced function, no explanation needed:')[1].split('A:')[0].strip()
    print(changed_function)
    data["changed_function"] = changed_function

outfile = open("../../alpha/deepseek_data_alpha.json", 'w')
outfile.write(json.dumps(data_res))
outfile.close()
