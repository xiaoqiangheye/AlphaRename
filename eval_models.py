from alpha.evaluate_metrics import start_evaluate
from alpha.evaluate_metrics import start_evaluate_substitution
import gemini.evaluate_alpha as alpha

f = open('model_evaluation_results_new.txt', 'w')

starcoder_results = start_evaluate('alpha/evaluation_data/starcoder_data_alpha_non_valid2.json')
print('starcoder: ', starcoder_results)
f.write('starcoder: ' + str(starcoder_results) + '\n')

deepseek_substitution_results = start_evaluate_substitution('/home/vl2395/AlphaRename/alpha/evaluation_data/deepseek_data_substitution_tasks.json')
print('deepseek substitution: ', deepseek_substitution_results)
f.write('deepseek substitution: ' + str(deepseek_substitution_results) + '\n')

deepseek_few_shot_results = start_evaluate('/home/vl2395/AlphaRename/alpha/evaluation_data/deepseek_data_alpha_non_valid2_few_shot.json')
print('deepseek few shot: ', deepseek_few_shot_results)
f.write('deepseek few shot: ' + str(deepseek_few_shot_results) + '\n')

llama_few_shot_results = alpha.evaluate("alpha/evaluation_data/llama_data_alpha_non_valid2_few_shot.json")
print('llama few shot: ', llama_few_shot_results)
f.write('llama few shot: ' + str(llama_few_shot_results) + '\n')
