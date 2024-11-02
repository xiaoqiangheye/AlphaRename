import requests
import json

API_URL = "https://api-inference.huggingface.co/models/bigcode/starcoder"
headers = {"Authorization": "Bearer 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"}

def query(payload):
	response = requests.post(API_URL, headers=headers, json=payload)
	return response.json()
	

output = query({
    "inputs": "Given a python function foo, we want to replace the parameter x with z and with semantics and logics preserved. Mark the start and the end of function with ##\n.\nHere is the function:\n\ndef foo(x: int):\n  return (lambda x: x + 1)(x) + (lambda x: x * 2)(x)\n\nHere is the replaced function, no explanation needed:\n\n##\ndef foo(",
})


changed_function = output[0]['generated_text'].split('##')[2]
print(changed_function)

