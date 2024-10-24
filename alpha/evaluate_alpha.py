

import vertexai
from vertexai.generative_models import GenerativeModel
import json

# Currently the alpha renaming tasks generated is a little bit simple. Hard to find existing dataset that
# is suitable for alpha renaming. One requirement is that the function code has multiple same name argument 
# but in different scope.

PROJECT_ID = "alpha-rename"
DATASET = "data.json"
vertexai.init(project=PROJECT_ID, location="us-central1")

model = GenerativeModel("gemini-1.5-flash-002")


prompt = '''You are a code programmer, we would like you to perform a alpha-renaming task a given function, 
changing a function argument name to another name and preserve the semantics. \n
That means bounded variables should not be changed even if the name is identical to the argument name. \n

Here is the function

``function begins``
'''


# read dataset
with open(DATASET) as f:
    dataset = json.load(f)
    for data in dataset:
        target_argu = data["target_argument"]
        change_ar  = data["change_to"]
        original_function = data["original_function"]
        expected = data["changed_function"]

        prompt_task = f'''{prompt}
                     {original_function}
                    ``function ends``

                    The argument we want you to change is '{target_argu}'
                    You need to change this argument to '{change_ar}' and preserve the semantics of the function.
                    Generate only the output as the following format, do not include any extra output,
                    do not include any comments. Only generate the output as the following format.
                    {{
                      changed_function : "put changed function here" 
                    }}
                    '''

        response = model.generate_content(prompt_task)
        text = response.text[8:len(response.text)-4]
        print(text)

        res_j = json.loads(text)
        changed_function = res_j["changed_function"]
        print("changed function:" + changed_function)
        
        ##TODO: run the changed function and expected functions and compare results
        

        








#response = model.generate_content("")