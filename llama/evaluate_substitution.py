import requests
import json
import alpha.evaluate_metrics as alpha
from transformers import AutoTokenizer
import transformers
import torch
from huggingface_hub import InferenceClient

<<<<<<< HEAD
model = "meta-llama/CodeLlama-7b-hf"
=======
model = "meta-llama/CodeLlama-7b-Instruct-hf"
>>>>>>> 4d09e250086b42d614ff6cf14b779dae1a2d57c9

tokenizer = AutoTokenizer.from_pretrained(model)
pipeline = transformers.pipeline(
        "text-generation",
        model=model,
        torch_dtype=torch.float16,
        device="cuda"
        )

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

def generate_function(input_text):
    torch.cuda.empty_cache()
    res = pipeline(prompt+input_text,do_sample=True,top_k=10,temperature=0.1,top_p=0.95,num_return_sequences=1,eos_token_id=tokenizer.eos_token_id, max_length=len(input_text)) 
<<<<<<< HEAD
#    print('\nres', res[0])
    changed_function = res[0]['generated_text'].split('``function begins``')[1].split('``function ends``')[0].strip()
=======
    print('\nres', res[0])
    changed_function = res[0]['generated_text']
>>>>>>> 4d09e250086b42d614ff6cf14b779dae1a2d57c9

    #print('changed', changed_function)
    return changed_function


def evaluate(DATASET = "./alpha/dataset/data_substitution_tasks.json"):
    f = open(DATASET, 'r')
    data_res = json.loads(f.read())

    total_accuracy = 0.0
    total_count = 0
    for data in data_res:
        total_count += 1
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

        changed_function = generate_function(input_text)
        data['changed_function'] = changed_function

    #    print("original", original_function)
        print("changed", changed_function)

        accuracy = alpha.evaluate_substitution(original_function, function_name, variable, expr, output_expr, inputs)
        total_accuracy += accuracy
<<<<<<< HEAD
        total_count += 1
=======
        print(total_accuracy/total_count)
>>>>>>> 4d09e250086b42d614ff6cf14b779dae1a2d57c9

    print("final accuracy: ", total_accuracy/total_count)
    outfile = open("./alpha/evaluation_data/llama_data_substitution_tasks.json", 'w')
    outfile.write(json.dumps(data_res))
    outfile.close()
