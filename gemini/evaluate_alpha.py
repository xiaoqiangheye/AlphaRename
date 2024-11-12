

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
prompt = '''You are a code programmer, we would like you to perform a alpha-renaming task a given function, 
changing a function argument name to another name and preserve the semantics. \n

Here is the function

``function begins``
'''

def evaluate(DATASET = "alpha/dataset/data_alpha_non_valid2.json"):
  # read dataset
  with open(DATASET) as f:
      dataset = json.load(f)
      accuracy, total_accuracy = 0.0, 0.0
      total_count = 0
      for data in dataset:
          total_count += 1
          target_argu = data["target_argument"]
          change_ar  = data["change_to"]
          original_function = data["original_function"]
          expected = data["changed_function"]
          name = data["function_name"]
          inputs = data["inputs"]

          prompt_task = f'''{prompt}
  {original_function}
  ``function ends``
  1.The argument we want you to change is '{target_argu}'
  2.You need to change this argument to '{change_ar}' and preserve the semantics of the function.
  3.Generate only the output as the following json format, do not include any extra output, do not include any comments. Only generate the output as the following format.
  {{
      changed_function : "put changed function here" 
    }}
  '''
          response = model.generate_content(prompt_task, generation_config=generation_config)
          print(response.text)

          res_j = json.loads(response.text)
          changed_function = res_j["changed_function"]
          print("changed function:" + changed_function)
          
          ##TODO: Verification method: run the changed function and expected functions and compare results
          accuracy = alpha.evaluate(original_function, changed_function, name, inputs)
          total_accuracy += accuracy

  print("final accuracy: ", total_accuracy/total_count)
  return total_accuracy, total_count