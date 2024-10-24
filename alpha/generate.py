import vertexai
from vertexai.generative_models import GenerativeModel

# TODO(developer): Update and un-comment below line
PROJECT_ID = "alpha-rename"
vertexai.init(project=PROJECT_ID, location="us-central1")

model = GenerativeModel("gemini-1.5-pro")


prompt = '''
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
{ target_argument : “f”,
  change_to: “a”
  original_function: “def foo (f: int)\n :
	                          (lambda f: f 1)(lambda y: y + f)”
  changed_function: “def foo (a: int)\n :
	                          (lambda f: f 1)(lambda y: y + a)”
}
‘’’Example end’’’

But, we need more complex functions that involve more lambda functions and more arguments with the same name but in different scopes.
1. Please generate function examples that are more than 5 lines, with local function definitions and lambda definitions, and each pair of local definitions must share at least one argument name.
2. Each function should only has one argument.
3. Gain inspiration from the hint text '{hint}'
'''
response = model.generate_content(

)




print(response.text)

# Example response:
# **Emphasizing the Dried Aspect:**
# * Everlasting Blooms
# * Dried & Delightful
# * The Petal Preserve
# ...
