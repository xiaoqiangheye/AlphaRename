import requests
import json
import alpha.evaluate_metrics as alpha

API_URL = "https://api-inference.huggingface.co/models/bigcode/starcoder"
headers = {"Authorization": "Bearer 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"}

with open("./starcoder/prompt_file.txt", "r") as f:
    prompt = f.read()+'\n'



with open("./starcoder/prompt_file_cot.txt", "r") as f:
    few_shot_cot = json.load(f)
    few_shot_cot_promt = ""
    for data in few_shot_cot:
        cot = data["step-by-step thoughts"]
        changed = data["changed_function"]
        original = data["original_function"]
        name = data["function_name"]
        argument = data["target_argument"]
        change_to = data["change_to"]
        few_shot_cot_promt += f'''Human: Given a python function {name} we want to replace the parameter '{argument}' with '{change_to}', with semantics and logics of the function preserved. 
Here is the function
{original}

Analyze the problems step by step and return the replaced function
Assistant:
{cot}

returned function:
{changed}
'''



def query(payload):
    response = requests.post(API_URL, headers=headers, json=payload)
    return response.json()



def generate_cot_function(original_function, function_name, argument_name, change_to):
    output = query({
        "inputs": f'''{few_shot_cot_promt}
Human: Given a python function {function_name} we want to replace the parameter '{argument_name}' with '{change_to}', with semantics and logics of the function preserved. 
Here is the function
{original_function}

Analyze the problems step by step and return the replaced function
Assistant:
'''
})

    changed_function = output[0]['generated_text']
    #print("output: ", changed_function)
    changed_function = changed_function.split('returned function:')[-1]
    changed_function = changed_function.split('Human:')[0].strip()
    #print("changed_function: ", changed_function)
    return changed_function


def generate_function(original_function, function_name,
        argument_name, change_to):
    output = query({
        "inputs": prompt + "Human: " + f"Given a python function '{function_name}', we want to replace the parameter '{argument_name}' with '{change_to}', with semantics and logics of the function preserved.\nHere is the function\n{original_function}\n" +  "Assistant:"
})
    changed_function = output[0]['generated_text'].split('Assistant:')[-1].split('-----')[0].strip()
    main, return_statement = 'return'.join(changed_function.split('return')[0:-1]), changed_function.split('return')[-1].split('\n')[0]
    changed_function = main+'return'+return_statement
    print(changed_function)
    return changed_function



def evaluate_cot(path='./alpha/dataset/data_alpha_non_valid2.json'):
    f = open(path, 'r')
    data_res = json.loads(f.read())
    total_count = 0
    total_accuracy = 0
    for data in data_res:
        total_count += 1
        argument_name = data["target_argument"]
        change_to = data["change_to"]
        # changed_function = data["changed_function"]
        original_function = data["original_function"]
        function_name = data["function_name"]
        inputs = data["inputs"]

        try:
            changed_function = generate_cot_function(
                    original_function, function_name, argument_name, change_to)
            data["changed_function"] = changed_function
        except:
            continue

        accuracy = alpha.evaluate(
                original_function, changed_function, function_name, inputs)
        print(accuracy)
        total_accuracy += accuracy
    
    outfile = open("./alpha/evaluation_data/starcoder_data_alpha_non_valid2_cot.json", 'w')
    outfile.write(json.dumps(data_res))
    outfile.close()
    print("final accuracy:", total_accuracy/total_count)
    return total_accuracy, total_count

def evaluate(path='./alpha/dataset/data_alpha_non_valid2.json'):
    f = open(path, 'r')
    data_res = json.loads(f.read())
    total_count = 0
    total_accuracy = 0
    for data in data_res:
        total_count += 1
        argument_name = data["target_argument"]
        change_to = data["change_to"]
        # changed_function = data["changed_function"]
        original_function = data["original_function"]
        function_name = data["function_name"]
        inputs = data["inputs"]

        changed_function = generate_function(
                original_function, function_name, argument_name, change_to)
        data["changed_function"] = changed_function

#        accuracy = alpha.evaluate(
 #               original_function, changed_function, function_name, inputs)
  #      total_accuracy += accuracy

    outfile = open("./alpha/evaluation_data/starcoder_data_alpha_non_valid2_few_shots.json", 'w')
    outfile.write(json.dumps(data_res))
    outfile.close()

   # print("final accuracy:", total_accuracy/total_count)
    return total_accuracy, total_count




