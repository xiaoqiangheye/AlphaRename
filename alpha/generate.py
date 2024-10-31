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
Generate dataset of pairs of original python function, a target argument name, a name to change original argument to, after-changed function, and function call expression with input 1, output with the following format.
Note that you only need to generate the dataset, not the alpha-renaming function. The dataset is with the following format.

\'\'\'format start\'\'\'
{ target_argument : “name of one argument needs to change”,
  change_to: “the argument name we want to change to”,
  original_function: “this is where to put original functions”,
  changed_function: “this is where to put the alpha-renamed one of the original functions”,
  function_call: "this is where to put the function call expression of the original function"
  
}
\'\'\'format ends\'\'\'

We also have an example output.

‘’’Example begin’’’
{ target_argument : “f”,
  change_to: “a”,
  original_function: “def foo (f: int):\n
                              a = 1\n
	                          return (lambda f: f 1)(lambda y: y + f)\n”,
  changed_function: “def foo (a: int):\n
                              b = 1\n
	                          return (lambda f: f 1)(lambda y: y + a)”,
  function call: "foo(1)"
}
‘’’Example end’’’

But, we need more complex functions that involve more lambda functions and more arguments with the same name but in different scopes.
1. Please generate function examples that are more than 5 lines with local function definitions or lambda definitions.
2. Each function should only has only one argument.
3. Gain inspiration from the hint text '{hint}'
4. Consider also common algorithms such sorting, dfs, bfs....from algorithm classes, implement them in resursive functional versions.
5. Once the argument is changed, you might need to check if it has conflict with other variables in scope, in that case you need to also rename other variables in conflict to fresh names to preserve semantics.
6. Output in Json format
7. Generate 5 example, each one has different function name. Output in a json list, [example1, example2...].
8. Ensure the Code is e
'''


datas = []
with open(HINT_FILE,"r") as f:
    hints = json.load(f)
    for hint in hints:
        response = model.generate_content(prompt(hints[1]), generation_config=generation_config)
        data_res = json.loads(response.text)
        for data in data_res:
            argument_name = data["target_argument"]
            changed_function = data["changed_function"]
            original_function = data["original_function"]
            change_to = data["change_to"]
            function_call = data["function_call"]

            print("argument_name: ",argument_name)
            print("changed_function", changed_function)
            print("original_function", original_function)
            print("change_to", change_to)
            print("function_call", function_call)
            
            ## we could also add verification here to ensure the generated code is executable and
            ## generate the right after change function. See generate_execution.py to use python interpreter.
            datas.append(data)

with open(OUTPUT_FILE, "w") as outfile:
    json.dump(datas, outfile)
