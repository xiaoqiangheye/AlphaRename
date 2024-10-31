import json

path = './data.json'
f = open(path, 'r')

valid_data = []
data = json.load(f)

for case in data:
    try:
        
        exec(case["original_function"])
        exec(case["changed_function"] )
        valid_data.append(case)
    except Exception as e:
       pass

nf = open('./valid_data.json', 'w')
nf.write(json.dumps(valid_data))
