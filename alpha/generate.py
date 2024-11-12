import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import json
from datasets import load_dataset

from datasets import load_dataset
from evaluate_metrics import evaluate

mbpp = load_dataset("google-research-datasets/mbpp", "full")


# TODO(developer): Update and un-comment below line
PROJECT_ID = "alpha-rename"
vertexai.init(project=PROJECT_ID, location="us-central1")
HINT_FILE = "./hints.json"
model = GenerativeModel("gemini-1.5-flash")
generation_config = GenerationConfig(
    temperature=1.5,
    top_k=40,
    response_mime_type="application/json"
   )

prompt = lambda num, hint: f'''
We would like to generate datasets for alpha-renaming tasks, changing a function argument name to another name and preserve the semantics. \n
That means bounded variables should not be changed even if the name is identical to the argument name. \n
Generate dataset of pairs of original python function, a target argument name, a name to change original argument to, and after-changed function, output with the following format.
Note that you only need to generate the dataset, not the alpha-renaming function. The dataset is with the following format.

\'\'\'format start\'\'\'
{{ "target_argument" : “name of one argument needs to change”,
  "change_to": “the argument name we want to change to”,
  "original_function": “this is where to put original functions”,
  "changed_function": “this is where to put the alpha-renamed one of the original functions”,
  "function_name" : "<function_name>",
  "inputs": ["<input1>", "<input2>"...]
}}
\'\'\'format ends\'\'\'

We also have an example output.

‘’’Example begin’’’
{{ "target_argument" : “f”,
  "change_to": “a”,
  "original_function": “def foo (f: int):\n
                              a = 1\n
	                          return (lambda f: f 1)(lambda y: y + f)\n”,\n
  "changed_function": “def foo (a: int):\n
                              b = 1\n
	                          return (lambda f: f 1)(lambda y: y + a)”,\n
  "function_name" : "foo", 
  "inputs": ["5", "10"...]
}}

‘’’Example end’’’

But, we need more complex functions.
1. Please generate function examples that are more than 5 lines.
2. Each function should only has only one argument, each arugment has base types such as bool, int, strings, or list or tuple of simple types.
3. Gain inspiration from the hint text
<<
{hint}
>>
4. For after-change funciton, you might need to check if it has conflict with other variables in scope, in that case you need to also rename other variables in conflict to fresh names to preserve semantics.
5. Output in Json format, very important
6. Generate {num} example, each one has different function name. Output in a json list, [example1, example2...].
7. Also generate a list of 5 valid inputs for each function as a json list. ["input1", "input2"...]. Input must be in python expressions that is quoted as string and can
directly be feeded into the function.
8. You should pick a after-change name that already exists in the functions to create name conflicts to make problem more complex. For example if you want to chagne x
to n, then n must already exists in the original function.
'''


NUM_SAMPLES = 500
EACH_TIME = 10
CHECK_VALID = False
OUTPUT_FILE = "data_alpha_non_valid_after_change_500.json"

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
                argument_name = data["target_argument"]
                changed_function = data["changed_function"]
                original_function = data["original_function"]
                change_to = data["change_to"]

                function_name = data["function_name"]
                inputs = data["inputs"]

                print("argument_name: ",argument_name)
                print("changed_function", changed_function)
                print("original_function", original_function)
                print("change_to", change_to)
                print("function_name", function_name)
                print("inputs", inputs)
                
                
                # in case the generated dataset is used for training, still validate the changed function generated.
                if(CHECK_VALID):
                    res = evaluate(original_function=original_function, changed_function=changed_function, function_name=function_name, inputs=inputs)
                    if res == 1.0:
                        print("valid data")
                        valid_data.append(data)
                else:
                    try:
                        original_program = original_function+"\n"+function_name+"("+inputs[0]+")"
                        changed_program  = changed_function +"\n"+function_name+"("+inputs[0]+")"
                        #only check execution of original program if check valid is not true
                        exec(original_program)
                        # exec(changed_program)
                        valid_data.append(data)
                    except Exception as e:
                        pass
        except Exception as e:
            print("bad json text")
            pass

with open(OUTPUT_FILE, "w") as outfile:
    json.dump(valid_data, outfile, indent=1)

print(len(valid_data), "valid data")