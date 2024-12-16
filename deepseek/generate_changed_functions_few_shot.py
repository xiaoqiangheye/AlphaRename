from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-instruct", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

path = '../alpha/dataset/data_alpha_non_valid2.json'
f = open(path, 'r')
data_res = json.loads(f.read())

prompt = '''You are a code programmer, we would like you to perform a alpha-renaming task a given function,
changing a function argument name to another name and preserve the semantics. \n
You probably need to rename other names in the functions to avoid name conflits. \n

Here is the function

``function begins``
'''

with open("./prompt_file.txt", "r") as f:
    messages = json.loads(f.read())['messages']

torch.cuda.empty_cache()
for data in data_res:

    argument_name = data["target_argument"]
    change_to = data["change_to"]
    changed_function = data["changed_function"]
    original_function = data["original_function"]
    function_name = data["function_name"]
    inputs = data["inputs"]

    input_text = {'role': 'user', 'content': f'''Given a python function {function_name} we want to replace the parameter '{argument_name}' with '{change_to}'. Here is the original function:\n\n{original_function}\n\n'''}
    inputs = tokenizer.apply_chat_template(messages+[input_text], add_generation_prompt=True, return_tensors="pt").to(model.device)
    model_output = model.generate(inputs, max_length=len(original_function)+800, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    changed_function = tokenizer.decode(model_output[0], skip_special_tokens=True).split('### Response:')[-1]
    print(changed_function)
    data["changed_function"] = changed_function

outfile = open("../alpha/evaluation_data/deepseek_data_alpha_non_valid2_few_shot.json", 'w')
outfile.write(json.dumps(data_res))
outfile.close()

