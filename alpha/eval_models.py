from evaluate_metrics import start_evaluate

f = open('model_evaluation_results.txt', 'w')

starcoder_results = start_evaluate('./starcoder_data_alpha.json')
print('starcoder: ', starcoder_results)
f.write('starcoder: ' + str(starcoder_results) + '\n')

deepseek_results = start_evaluate('./deepseek_data_alpha.json')
print('deepseek: ', deepseek_results)
f.write('deepseek: ' + str(deepseek_results) + '\n')
