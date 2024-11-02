import json

path = './data_alpha.json'
f = open(path, 'r')

data_res = json.loads(f.read())

def evaluate(original_function, changed_function, function_name, inputs):
    accurate_results = 0
    try: 
        for val in inputs:
            function_call = function_name+"("+val+")"
            original_program = original_function + "\n" + function_call
            changed_program  = changed_function + "\n" + function_call

            original_output = exec(original_program)
            changed_output  = exec(changed_program)

            if original_output == changed_output: accurate_results+=1

        return accurate_results/len(inputs)
    except Exception as e:
        return 0

total_count, total_accuracy = 0, 0

for data in data_res:

    total_count +=1 

    argument_name = data["target_argument"]
    changed_function = data["changed_function"]
    original_function = data["original_function"]
    change_to = data["change_to"]
    function_call = data["function_call"]
    function_name = data["function_name"]
    inputs = data["inputs"]
    
    accuracy = evaluate(original_function, changed_function, function_name, inputs)
    total_accuracy += accuracy

    print(total_count, accuracy)

print("final accuracy: ", total_accuracy/total_count)