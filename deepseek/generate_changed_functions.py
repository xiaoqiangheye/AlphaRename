from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

path = '../../alpha/data_alpha.json'
f = open(path, 'r')
data_res = json.loads(f.read())

prompt = '''You are a code programmer, we would like you to perform a alpha-renaming task a given function,
changing a function argument name to another name and preserve the semantics. \n
You probably need to rename other names in the functions to avoid name conflits. \n

Here is the function

``function begins``
'''

for data in data_res:

    argument_name = data["target_argument"]
    change_to = data["change_to"]
    changed_function = data["changed_function"]
    original_function = data["original_function"]
    function_name = data["function_name"]
    inputs = data["inputs"]

    prompt_task = f'''{prompt}
  {original_function}
  ``function ends``
  1.The argument we want you to change is '{argument_name}'
  2.You need to change this argument to '{change_to}' and preserve the semantics of the function.
  3.Generate only the output as the following json format, do not include any extra output, do not include any comments. Only generate the output as the following format.
  {{
      changed_function : "put changed function here" 
    }}
  '''

    #input_text = "Given a python function " +function_name + ", we want to replace the parameter " + argument_name +" with " + change_to + " and with semantics and logics preserved.\nHere is the function:\n\n" + original_function
    model_input = tokenizer(prompt_task, return_tensors="pt").to(model.device)
    model_output = model.generate(**model_input, max_length=128)
    changed_function = tokenizer.decode(model_output[0], skip_special_tokens=True)
    print(changed_function)
    data["changed_function"] = changed_function

outfile = open("../../alpha/starcoder_data_deepseek.json", 'w')
outfile.write(json.dumps(data_res))
outfile.close()

