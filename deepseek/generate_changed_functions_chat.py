from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

path = '../alpha/dataset/data_alpha_non_valid2.json'
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

    target_argument = data["target_argument"]
    change_to = data["change_to"]
    changed_function = data["changed_function"]
    original_function = data["original_function"]
    function_name = data["function_name"]
    inputs = data["inputs"]

    messages=[
    { 'role': 'user', 'content': f'''{prompt}
  {original_function}
  ``function ends``
  1.The argument we want you to change is '{target_argument}'
  2.You need to change this argument to '{change_to}' and preserve the semantics of the function.
  3.Generate only the output as the following json format, do not include any extra output, do not include any comments. Only generate the output as the following format.
  '''}]
    model_input = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt").to(model.device)
    model_output = model.generate(model_input, max_new_tokens=512, do_sample=False, top_k=50, top_p=0.95, num_return_sequences=1, eos_token_id=tokenizer.eos_token_id)
    output = tokenizer.decode(model_output[0], skip_special_tokens=True)
    changed_function = output.strip().split('``function begins``')[2].split('``function ends``')[0]
    print(changed_function)
    data["changed_function"] = changed_function

outfile = open("../alpha/evaluation_data/deepseek_chat_data_alpha.json", 'w')
outfile.write(json.dumps(data_res))
outfile.close()

