import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import json

# TODO(developer): Update and un-comment below line
PROJECT_ID = "alpha-rename"
vertexai.init(project=PROJECT_ID, location="us-central1")
HINT_FILE = "hints.json"
model = GenerativeModel("gemini-1.5-flash")
generation_config = GenerationConfig(
    temperature=1.5,
    top_k=40,
    response_mime_type="application/json"
   )

prompt = lambda hint: '''
We would like to generate datasets for alpha-renaming tasks, changing a function argument name to another name and preserve the semantics. \n
That means bounded variables should not be changed even if the name is identical to the argument name. \n
Generate dataset of pairs of original python function, a target argument name, a name to change original argument to, and after-changed function, output with the following format.
Note that you only need to generate the dataset, not the alpha-renaming function. The dataset is with the following format.

\'\'\'format start\'\'\'
{ target_argument : “name of one argument needs to change”,
  change_to: “the argument name we want to change to”
  original_function: “this is where to put original functions”
  changed_function: “this is where to put the alpha-renamed one of the original functions”
}
\'\'\'format ends\'\'\'

We also have an example output.

‘’’Example begin’’’
{ "target_argument" : “f”,
  "change_to": “a”,
  "original_function": “def foo (f: int):\n
                              a = 1\n
	                          return (lambda f: f 1)(lambda y: y + f)\n”,
  "changed_function": “def foo (a: int):\n
                              b = 1\n
	                          return (lambda f: f 1)(lambda y: y + a)”,
  "function_name" : "foo", 
  "inputs": ["input1", "input2"...]
}

‘’’Example end’’’

But, we need more complex functions that involve more lambda functions and more arguments with the same name but in different scopes.
1. Please generate function examples that are more than 5 lines.
2. Each function should only has only one argument, each arugment has simple types such as bool, int, strings, list of simple types.
3. Gain inspiration from the hint text '{hint}'
4. Once the argument is changed, you might need to check if it has conflict with other variables in scope, in that case you need to also rename other variables in conflict to fresh names to preserve semantics.
5. Output in Json format
6. Generate 5 example, each one has different function name. Output in a json list, [example1, example2...].
7. Also generate a list of 5 valid inputs for each function as a json list, ["input1", "input2"...]. The inputs should be simple inputs like int, bool, strings, tuples of base types or list of base types.
   each input should be quoted as strings and adhere to python language and can be directly pass to a python function.
8. there should be conflict of sames names for the changed_name and existing names in the function in different scope to make problems more complex.
'''


NUM_SAMPLES = 5
EACH_TIME = 5
OUTPUT_FILE = "data_alpha.json"

iterations = NUM_SAMPLES // EACH_TIME
datas = []
with open(HINT_FILE,"r") as f:
    hints = json.load(f)
    for i in range(iterations):
        response = model.generate_content(prompt(hints[i]), generation_config=generation_config)
        print(response.text)
        data_res = json.loads(response.text)
        for data in data_res:
            argument_name = data["target_argument"]
            changed_function = data["changed_function"]
            original_function = data["original_function"]
            change_to = data["change_to"]
            functiton_name = data["function_name"]
            inputs = data["inputs"]

            print("argument_name: ",argument_name)
            print("changed_function", changed_function)
            print("original_function", original_function)
            print("change_to", change_to)
            print("function_name", functiton_name)
            print("inputs", inputs)
            ## we could also add verification here to ensure the generated code is executable and
            ## generate the right after change function. See generate_execution.py to use python interpreter.
            for inp in inputs:
                command = f'''
out = {functiton_name}({inp})
print(out)
                
'''
            datas.append(data)

with open(OUTPUT_FILE, "w") as outfile:
    json.dump(datas, outfile)
