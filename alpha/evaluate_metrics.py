import json
import io
from contextlib import redirect_stdout
import signal


def evaluate_substitution(original_function, function_name, argument, expr, output_expr, inputs):
    accurate_results = 0
    try:
        for val in inputs:
            original_program = original_function+"\n"+"print("+function_name+f"((lambda {argument}:{expr})({val})))"
            changed_program = f'''
{argument} = {val}
{output_expr}
print({function_name}())
'''
            
            changed_buffer = io.StringIO()
            with redirect_stdout(changed_buffer):
                exec(changed_program,{})
            changed_output = changed_buffer.getvalue().strip()

            original_buffer = io.StringIO()
            with redirect_stdout(original_buffer):
                exec(original_program,{})
            original_output = original_buffer.getvalue().strip()

            if original_output == changed_output: accurate_results+=1
        return accurate_results==len(inputs)
    except Exception as e:
        print(e)
        return 0

    
def evaluate(original_function, changed_function, function_name, inputs):
    accurate_results = 0
    try: 
        for val in inputs:
            function_call = function_name+"("+val+")"
            original_program = f'''

import signal
{original_function}

def timeout_handler(signum, frame):
    signal.alarm(0)
    raise TimeoutError("time out")
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(5)

try:    
    print({function_call})
except Exception:
    print("error")
finally:
    signal.alarm(0)
'''
            
            changed_program  = f'''
import signal

{changed_function}

def timeout_handler(signum, frame):
   signal.alarm(0)
   raise TimeoutError("time out")
signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(5)
try:
    print({function_call})
except Exception:
    print("error")
finally:
    signal.alarm(0) 
'''
            print(changed_program)
           
            
            changed_buffer = io.StringIO()
            try:
                original_buffer = io.StringIO()
                with redirect_stdout(original_buffer):
                    exec(original_program,{})
                original_output = original_buffer.getvalue().strip()
                with redirect_stdout(changed_buffer):
                    exec(changed_program,{})
                changed_output = changed_buffer.getvalue().strip()
            except TimeoutError as e:
                print(e)
                return 0

            #print(val, original_output, changed_output)
            
            if original_output == changed_output: accurate_results+=1
            #else: print(original_output, changed_output)
        print(accurate_results/len(inputs))
        return accurate_results==len(inputs)
    except Exception as e:
        print(e)
        return 0



def start_evaluate(path="alpha/deepseek_data_alpha.json"):

    with open(path, 'r') as f:
        data_res = json.loads(f.read())

    total_count, total_accuracy = 0, 0
    for data in data_res:
        total_count +=1 
        argument_name = data["target_argument"]
        changed_function = data["changed_function"]
        original_function = data["original_function"]
        change_to = data["change_to"]
        function_name = data["function_name"]
        inputs = data["inputs"]
        accuracy = evaluate(original_function.strip(), changed_function.strip(), function_name, inputs)
        total_accuracy += accuracy 

        print(total_count, accuracy)

    print("final accuracy: ", total_accuracy/total_count)
    return total_accuracy/total_count

def start_evaluate_substitution(path="alpha/deepseek_data_alpha.json"):

    with open(path, 'r') as f:
        data_res = json.loads(f.read())

    total_count, total_accuracy = 0, 0
    for data in data_res:
        total_count +=1 
        expr = data["expr"]
        variable = data["variable"]
        original_function = data["original_function"]
        output_expr = data["output_expr"]
        function_name = data["function_name"]
        inputs = data["inputs"]
        changed_function = data["changed_function"]

        accuracy = evaluate_substitution(original_function.strip(), function_name, variable, expr, output_expr, inputs)
        total_accuracy += accuracy 

        print(total_count, accuracy)

    print("final accuracy: ", total_accuracy/total_count)
    return total_accuracy/total_count


if __name__ == "__main__":
    start_evaluate()
