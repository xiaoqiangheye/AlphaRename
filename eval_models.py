from alpha.evaluate_metrics import start_evaluate
import gemini.evaluate_alpha as alpha

f = open('model_evaluation_results_new.txt', 'w')

starcoder_results = start_evaluate('alpha/evaluation_data/starcoder_data_alpha_non_valid2.json')
print('starcoder: ', starcoder_results)
f.write('starcoder: ' + str(starcoder_results) + '\n')

deepseek_results = start_evaluate('alpha/evaluation_data/deepseek_data_alpha_non_valid2.json')
print('deepseek: ', deepseek_results)
f.write('deepseek: ' + str(deepseek_results) + '\n')


#gemini_results = alpha.evaluate("alpha/data_alpha_non_valid_after_change_500.json")
#print('gemini: ', gemini_results)
#f.write('gemini: ' + str(gemini_results) + '\n')
