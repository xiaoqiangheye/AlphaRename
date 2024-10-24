import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import json
import io

PROJECT_ID = "alpha-rename"
generation_config = GenerationConfig(
    temperature=1.5,
    top_k=40
)
vertexai.init(project=PROJECT_ID, location="us-central1")

model = GenerativeModel("gemini-1.5-pro")

prompt = '''
You are a functional programming lover,
recall common function names and code snippet used in functional programming such as Ocaml and Haskell, one example is map, generate another 100 ones.
For function names, only generate the name itself.
For code snippet, generate self-contained code that only use standard liabrary of the languages.
Each text is formated as json format {{[text1, text2, text3...]}}
One example format is 
{{["foo", "foldl (+) 0 [1,2,3]"]}}
'''

response = model.generate_content(prompt, generation_config=generation_config)
print(response.text)
