
import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import json
import alpha.evaluate_metrics as alpha

PROJECT_ID = "alpha-rename"
vertexai.init(project=PROJECT_ID, location="us-central1")
model = GenerativeModel("gemini-1.5-flash")
generation_config = GenerationConfig(
    temperature=1.0,
    response_mime_type="application/json"
   )
prompt = '''We would like to perform one step of symbolic execution task. The task is to substitute the argument of function
with input expression and output the substituted body. This step correspond symbolic function application that applying a function
to an symbolic expression. The expression contains only one open variable left undefined. Here is an example, it output an function that has no argument but is equivalent to applying original function
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



def evaluate(DATASET = "alpha/dataset/data_substitution_tasks.json"):
  # read dataset
  with open(DATASET) as f:
      dataset = json.load(f)
      accuracy, total_accuracy = 0.0, 0.0
      total_count = 0
      for data in dataset:
          total_count += 1
          expr = data["expr"]
          variable  = data["variable"]
          original_function = data["original_function"]
          #output_expr = data["output_expr"]
          name = data["function_name"]
          inputs = data["inputs"]

          prompt_task = f'''{prompt}
  {original_function}
  ``function ends``
  Output the changed function with no arguments that has substitute the arguments with {expr}.
  The expression has only one variable, which is {variable}.
  Generate only the output as the following json format, do not include any extra output, do not include any comments. Only generate the output as the following format.
  {{
      "output_expr" : "put changed function here" 
  }}
  '''
          response = model.generate_content(prompt_task, generation_config=generation_config)
          print(response.text)

          res_j = json.loads(response.text)
          output_expr = res_j["output_expr"]
          #print("changed function:\n" + output_expr)
          accuracy = alpha.evaluate_substitution(original_function, name, variable, expr, output_expr, inputs)
          print(accuracy)
          total_accuracy += accuracy

  print("final accuracy: ", total_accuracy/total_count)
  return total_accuracy, total_count