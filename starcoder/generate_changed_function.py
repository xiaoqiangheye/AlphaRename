import requests
import json

API_URL = "https://api-inference.huggingface.co/models/bigcode/starcoder"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	
def generate_function(original_function, function_name, argument_name, change_to):
	output = query({
		"inputs": "Given a python function " +function_name + ", we want to replace the parameter " + argument_name +" with " + change_to + " and with semantics and logics preserved. Mark the start and the end of function with ##\n.\nHere is the function:\n\n" + original_function + "\n\nHere is the replaced function, no explanation needed:\n\n##\ndef " + function_name + "(" + change_to,
	})
	# print(output[0]['generated_text']+"\n")
	changed_function = output[0]['generated_text'].split('##')[2]
	print(changed_function)
	return changed_function

path = '../alpha/data_alpha.json'
f = open(path, 'r')
data_res = json.loads(f.read())

for data in data_res:

	argument_name = data["target_argument"]
	change_to = data["change_to"]
	changed_function = data["changed_function"]
	original_function = data["original_function"]
	function_call = data["function_call"]
	function_name = data["function_name"]
	inputs = data["inputs"]

	changed_function = generate_function(original_function, function_name, argument_name, change_to)
	data["changed_function"] = changed_function


outfile = open("../alpha/starcoder_data_alpha.json", 'w')
outfile.write(json.dumps(data_res))
outfile.close()
