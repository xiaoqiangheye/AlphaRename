
import requests
import json
import alpha.evaluate_metrics as alpha

API_URL = "https://api-inference.huggingface.co/models/bigcode/starcoder"
headers = {"Authorization": "Bearer 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"}



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

def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()

def generate_function(original_function, function_name,
        expr, variable):
    output = query({
        "inputs": prompt + f'''
{original_function}
``function ends``
Substitute the function argument with the expression {expr} and generate the new function
without argument that is equivalent of applying original function with the expression.
Be sure to rename variables if the variables in the expression conflict with the function local variables.

Here is the replaced function, no explanation needed:
``function begins``
def {function_name}():'''
        })
    print(output[0]['generated_text'])
    changed_function = output[0]['generated_text'].split("Here is the replaced function, no explanation needed:")[1].split('``function begins``')[1].split('``function ends``')[0]
    print(changed_function)
    return changed_function


def evaluate(path='./alpha/dataset/data_substitution_tasks.json'):
    f = open(path, 'r')
    data_res = json.loads(f.read())
    total_count = 0
    total_accuracy = 0
    for data in data_res:
        total_count += 1
        variable = data["variable"]
        expr = data["expr"]
        
        # changed_function = data["changed_function"]
        original_function = data["original_function"]
        function_name = data["function_name"]
        inputs = data["inputs"]

        changed_function = generate_function(
                original_function, function_name, expr, variable)
        data["output_expr"] = changed_function

        accuracy = alpha.evaluate_substitution(
               original_function, function_name, variable, expr, changed_function, inputs)
        print(accuracy)
        print("accuracy:", total_accuracy/total_count)
        total_accuracy += accuracy

    outfile = open("./alpha/evaluation_data/starcoder_substitution.json", 'w')
    outfile.write(json.dumps(data_res))
    outfile.close()

    print("final accuracy:", total_accuracy/total_count)
    return total_accuracy, total_count