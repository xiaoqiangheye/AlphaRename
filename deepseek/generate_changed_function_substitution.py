from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import json

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-coder-6.7b-base", trust_remote_code=True, torch_dtype=torch.bfloat16).cuda()

path = '../alpha/dataset/data_substitution_tasks.json'
f = open(path, 'r')
data_res = json.loads(f.read())

prompt = '''We would like to perform one step of symbolic execution task. The task is to substitute the argument of function
with input expression and output the substituted body. This step correspond symbolic function application that applying a function
to an symbolic expression. Here is an example, it output an function that has no argument but is equivalent to applying original function
with the input expressions.

‘’’Example begin’’’
{{ 
   "expr": “x + 2”,
   "variable" : "x",
   "original_function": “def foo (y: int):
                              x = 1
	                          return y + x + 1”,
   "output_expr": “def foo():
                    z = 1
	                return x + 2 + z + 1”,
   "function_name" : "foo", 
   "inputs": ["5", "4", "1"]
}}

‘’’Example end’’’

Here is the function

``function begins``
'''

torch.cuda.empty_cache()
for data in data_res:

    expr = data["expr"]
    variable = data["variable"]
    original_function = data["original_function"]
    output_expr = data["output_expr"]
    function_name = data["function_name"]
    inputs = data["inputs"]

    input_text = f'''
{original_function}
``function ends``
Substitute the function argument with the expression {expr} and generate the new function
without argument that is equivalent of applying original function with the expression.
Be sure to rename variables if the variables in the expression conflict with the function local variables.

Here is the replaced function, no explanation needed:
``function begins``
def {function_name}():'''

    model_input = tokenizer(input_text, return_tensors="pt").to(model.device)
    model_output = model.generate(**model_input, max_length=len(original_function)+100)
    changed_function = tokenizer.decode(model_output[0], skip_special_tokens=True).split('``function begins``')[1].split('``function ends``')[0].strip()
    print(changed_function)
    data["changed_function"] = changed_function

outfile = open("../alpha/evaluation_data/deepseek_data_substitution_tasks.json", 'w')
outfile.write(json.dumps(data_res))
outfile.close()
