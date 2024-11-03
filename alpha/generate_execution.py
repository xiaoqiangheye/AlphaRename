import vertexai
from vertexai.generative_models import GenerativeModel, GenerationConfig
import json
import io
from contextlib import redirect_stdout

#Generate functions and its execution traces as substutions
#Problems: This is more like a training dataset, becaucase substution traces are not easy to be tested in a benchmark
#It is more like a intructional dataset suitable for finetuning


PROJECT_ID = "alpha-rename"
HINT = "hints.json"

prompt = lambda hint: f'''
We would like to generate instructional dataset for code executions tasks that involve
subsitution and higher order lambda functions passing around. 

The dataset is a pair of code and small-step style execution traces with step of step explainations on how
each step is tranformed.

\'\'\'format start\'\'\'
{{  function_name: "function name",
   function: “this is where to put original functions”,
   command : foo(1), 
   traces: “this is where to put the of the original functions”
   output : output based on the traces that you reasoned
}}
\'\'\'format ends\'\'\'

We also have an example output.

‘’’Example begin’’’
{{ 
  function_name: "foo",
  function: “def foo (f: int)\n :
	                          (lambda f: f 1)(lambda y: y + f)”
  command : "foo(1)",
  traces: "1.foo(1) -> (lambda f: f 1)(lambda y: y + 1)(1).  //Unfold the function definition\n
           2.(lambda f: f 1)(lambda y: y + f)(1) -> (lambda f: f 1)(lambda y: y + 1). //Since f in outer lambda functions is captured by lamda variable, it will
          not be substituted in this step. Only f on the right side will be substituted by 1.\n
           3. (lambda f: f 1)(lambda y: y + 1) -> (lambda y: y + 1) (1) //the inner lambda function is applied to outer lambda functions,
          thus f is substituted with the inner function.\n
           4. (lambda y: y + 1) (1) -> 1 + 1 //y is directly substitute with 1\n
           5. 1 + 1 = 2 // arithematic calculations\n
          ",
   ouput: 2
}}
‘’’Example end’’’

But, we need more complex functions. Here are some guidelines.
1.Please generate function examples that are more than 5 lines, with local function definitions and lambda definitions. 
2.Gain inspirations from the hints text [{hint}] from other functional languages.
3.Consider also common algorithms such sorting, dfs, bfs....from algorithm classes, implement them in resursive functional versions.
4.Combine those function to form random larger programs.
5.Also include other functions that you learn from functional programming, even rarely used ones.
6.Also come up with functions yourself that suits our purpose. 
7.Only generate self-contained functions that does not use external libraries and is implemented from scrach.
8. The main tasks is to generate traces complex enough to train an LLM. 
9. Please input human-reable comments in each step of traces to serve as explanations. However, do not include comments in function code.
10. For traces, please generate full traces and don't skip any steps. If traces length is larger than 1000 chracters, randomly delete some steps to make it less than 1000 characters.
11. Generate 5 samples, do not duplicate code.
Output all samples in a json list as [sample1, sample2, sample3...].
For commands, be sure to include the command with arguments such that it can be executed when the function is defined.
'''

NUM_SAMPLES = 500
EACH_TIME = 5

num_times = NUM_SAMPLES // EACH_TIME

assert(len(hints) >= num_times)
filtered = []
OUTPUT_FILE = "data_small_step500.json"


def execution_data_viewer():
   with open(OUTPUT_FILE, "r") as outfile:
      dataset = json.load(outfile)
      for data in dataset:
         func_def = data["function"]
         command = data["command"]
         output = data["output"]
         traces = data["traces"]
         name = data["function_name"]


def set_up_model():
   generation_config = GenerationConfig(
    temperature=1.5,
    top_k=40,
    response_mime_type="application/json"
   )
   vertexai.init(project=PROJECT_ID, location="us-central1")
   model = GenerativeModel("gemini-1.5-flash")
   return model
   
if __name__ == "__main__":
   model = set_up_model()
   with open(HINT) as f:
      hints = json.load(f)
      for i in range(num_times):
         response = model.generate_content(prompt(hints[i]), generation_config=generation_config)
         data = response.text
         print(data)
         try:
            data_json = json.loads(data)
         except:
            continue
         for data in data_json:
            try:
               func_def = data["function"]
               command = data["command"]
               output = data["output"]
               traces = data["traces"]
               name = data["function_name"]

               print(func_def)
               print(command)
               print(output)

               # Create a buffer to capture the output
               output_buffer = io.StringIO()
   

               code = f'''
{func_def}
out = {command}
print(out)
'''
               print(code)
               with redirect_stdout(output_buffer):
                  exec(code) #use python interpreter to ensure the code is executable.

               out = output_buffer.getvalue()
               print("output:" + out.strip(' \n\r'))
               print("expected:" + str(output))
               if(out.strip(' \n\r') == str(output)):
                  filtered.append(data)
            except:
               #errors, ignore the cases that unexecutable, or bad format.
               pass


   with open(OUTPUT_FILE, "w") as outfile:
      json.dump(filtered, outfile)