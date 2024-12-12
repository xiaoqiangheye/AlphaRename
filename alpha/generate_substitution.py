import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import json
from datasets import load_dataset

from evaluate_metrics import evaluate

mbpp = load_dataset("google-research-datasets/mbpp", "full")

PROJECT_ID = "alpha-rename"
vertexai.init(project=PROJECT_ID, location="us-central1")
HINT_FILE = "alpha/hints.json"
model = GenerativeModel("gemini-1.5-flash")
generation_config = GenerationConfig(
    temperature=1.5,
    top_k=40,
    response_mime_type="application/json"
   )

prompt = lambda num, hint: f'''
We would like to generate datasets for one step of symbolic execution task, a function takes an argument with type T will be supplied with a open term of type T that is not
fully evaluated value, so it may contains variables, function applications, ...etc. The task is to substitute the argument of function
with this expression and output the substituted body. This step correspond symbolic function application that applying a function
to an symbolic expressions. The task also requires one to resolve the name conflicts between the open expressions and the variables inside the functions.
Generate dataset of pairs of original python function, a target argument name, an open expression, and the expression after substitution, output with the following format.
Note that you only need to generate the dataset, not the alpha-renaming function. The dataset is with the following format.

\'\'\'format start\'\'\'
{{
  "expr": “the expr we want to substituted in”,
  "variable": "the only one variabe in expr",
  "original_function": “this is where to put original functions”,
  "output_expr": “this is where to put the output expressions”,
  "function_name" : "<function_name>",
  "inputs": ["<input1>", "<input2>"...]
}}
\'\'\'format ends\'\'\'

We also have an example output. Notice function foo has a local variable name x, and expression also has an open variable x.

‘’’Example begin’’’
{{ 
   "expr": “x + 2”,
   "variable" : "x",
   "original_function": “def foo (y: int):
                              x = 1
	                          return y + x + 1”,
   "output_expr": “z = 1
	               return x + 2 + z + 1”,
   "function_name" : "foo", 
   "inputs": ["5", "4", "1"]
}}

‘’’Example end’’’

But, we need more complex functions.
1. Please generate function examples that are more than 5 lines.
2. Each function should only has only one argument, each arugment has base types such as bool, int, strings, or list or tuple of simple types.
3. Each expression substitued in should be simple arithmatic expressions with only one variable. The expression should be self-contained.
The expression must match the type of function argument.
3. Gain inspiration from the hint text
<<
{hint}
>>
4. You might need to check if the expressions to be substituted in has conflict with local variables of functions, in that case you need to also rename local variables in conflict to fresh names to preserve semantics.
5. Output in Json format, very important
6. Generate {num} example, each one has different function name. Output in a json list, [example1, example2...].
7. Also generate a list of 5 valid values for the open variable as a json list. ["value1", "value2"...]. 
Each environment is a dictionary that map the open variable to its value. Input must be in python expressions that is quoted as string and can
directly be feeded into the function. 
8. You should pick an expression that contains variable name that already exists in the origin function to create name conflicts to make problem more complex.
'''


NUM_SAMPLES = 200
EACH_TIME = 5
CHECK_VALID = False
OUTPUT_FILE = "data_substitution_tasks.json"

iterations = NUM_SAMPLES // EACH_TIME
valid_data = []
with open(HINT_FILE,"r") as f:
    hints = json.load(f)
    for i in range(iterations):
        print("prompt", prompt(EACH_TIME, mbpp["test"][i]["code"]))
        response = model.generate_content(prompt(EACH_TIME, mbpp["test"][i]["code"]), generation_config=generation_config)
        print(response.text)
        try:
            data_res = json.loads(response.text)
            for data in data_res:
                argument_name = data["variable"]
                output_expr = data["output_expr"]
                original_function = data["original_function"]
                expr = data["expr"]

                function_name = data["function_name"]
                inputs = data["inputs"]

                print("argument_name: ",argument_name)
                print("output_expr", output_expr)
                print("original_function", original_function)
                print("expr", expr)
                print("function_name", function_name)
                print("inputs", inputs)
                
                #in case the generated dataset is used for training, still validate the changed function generated.
                if(CHECK_VALID):
                    res = evaluate_substitution(original_function=original_function, function_name=function_name, expr=expr, output_expr=output_expr, inputs=inputs)
                    if res == 1.0:
                        print("valid data")
                        valid_data.append(data)
                else:
                    try:
                        original_program = original_function+"\n"+"print("+function_name+f"((lambda {argument_name}:{expr})({inputs[0]})))"
                        print(original_program)
                        #changed_program  = changed_function +"\n"+function_name+"("+inputs[0]+")"
                        #only check execution of original program if check valid is not true
                        exec(original_program)
                        # exec(changed_program)
                        valid_data.append(data)
                    except Exception as e:
                        pass
        except Exception as e:
            print(e)
            print("bad json text")
            pass

with open(OUTPUT_FILE, "w") as outfile:
    json.dump(valid_data, outfile, indent=1)

print(len(valid_data), "valid data")