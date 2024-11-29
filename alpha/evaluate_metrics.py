import json
import io
from contextlib import redirect_stdout
import signal

def timeout_handler(signum, frame):
    raise TimeoutError("Function timed out")

def evaluate(original_function, changed_function, function_name, inputs):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(5)
    
    accurate_results = 0
    try: 
        for val in inputs:
            function_call = function_name+"("+val+")"
            original_program = original_function + "\nprint(" + function_call +')'
            changed_program  = changed_function + "\nprint(" + function_call +')'

            original_buffer = io.StringIO()
            with redirect_stdout(original_buffer):
                exec(original_program)
            original_output = original_buffer.getvalue().strip()
            
            changed_buffer = io.StringIO()
            try:
                with redirect_stdout(changed_buffer):
                    exec(changed_program)
                changed_output = changed_buffer.getvalue().strip()
            except TimeoutError as e:
                print(e)
                return 0

            #print(val, original_output, changed_output)
            
            if original_output == changed_output: accurate_results+=1
            #else: print(original_output, changed_output)
        print(accurate_results/len(inputs))
        return accurate_results/len(inputs)
    except Exception as e:
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



if __name__ == "__main__":
    start_evaluate()
