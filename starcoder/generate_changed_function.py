import requests
import json
import alpha.evaluate_metrics as alpha

API_URL = "https://api-inference.huggingface.co/models/bigcode/starcoder"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
def generate_function(original_function, function_name, argument_name, change_to):
	output = query({
		"inputs": "Given a python function " +function_name + ", we want to replace the parameter " + argument_name +" with " + change_to + " and with semantics and logics preserved. Mark the start and the end of function with ¥¥¥\n.\nHere is the function:\n\n" + original_function + "\n\nHere is the replaced function, no explanation needed:\n\n¥¥¥\ndef " + function_name + "(" + change_to,
	})
	# print(output[0]['generated_text']+"\n")
	changed_function = output[0]['generated_text'].split('¥¥¥')[2].split('\n\n')[0]
	print(changed_function)
	return changed_function

def evaluate(path = './alpha/dataset/data_alpha_non_valid2.json'):
	f = open(path, 'r')
	data_res = json.loads(f.read())
	total_count = 0
	total_accuracy = 0
	for data in data_res:
		total_count += 1
		argument_name = data["target_argument"]
		change_to = data["change_to"]
		#changed_function = data["changed_function"]
		original_function = data["original_function"]
		function_name = data["function_name"]
		inputs = data["inputs"]

		changed_function = generate_function(original_function, function_name, argument_name, change_to)
        data["changed_function"] = changed_function


		accuracy = alpha.evaluate(original_function, changed_function, function_name, inputs)
		total_accuracy += accuracy
	
    outfile = open("../alpha/evaluation_data/starcoder_data_alpha_non_valid2.json", 'w')
    outfile.write(json.dumps(data_res))
    outfile.close()
	
    print("final accuracy:", total_accuracy/total_count)
	return total_accuracy, total_count


	

