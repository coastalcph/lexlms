import json
import os
import argparse
import warnings
from data import DATA_DIR
import copy
warnings.filterwarnings('ignore')


def main():
    ''' set default hyperparams in default_hyperparams.py '''
    parser = argparse.ArgumentParser()

    # Required arguments
    parser.add_argument('--dataset',  default='ecthr_b')
    parser.add_argument('--filter_outliers', default=True)
    parser.add_argument('--model_name', default="lexlms/hi-transformer-base")
    config = parser.parse_args()

    BASE_DIR = f'{DATA_DIR}/lexglue_logs/{config.dataset}'

    if os.path.exists(BASE_DIR):
        print(f'{BASE_DIR} exists!')

    score_dict = {'dev': {'micro': [], 'macro': []},
                  'test': {'micro': [], 'macro': []},
                  'params': []}

    print(f'{"PARAMETERS":<20} {"VALIDATION":<47} | {"TEST"}')
    print('-' * 200)
    for pooling_method in ['attentive', 'max', 'first-cls']:
        dir_name = f'pooling_{pooling_method}'
        try:
            with open(os.path.join(BASE_DIR, config.model_name, dir_name, 'all_results.json')) as json_file:
                 json_data = json.load(json_file)
                 score_dict['dev']['micro'].append(float(json_data['eval_micro-f1']))
                 score_dict['dev']['macro'].append(float(json_data['eval_macro-f1']))
                 score_dict['test']['micro'].append(float(json_data['predict_micro-f1']))
                 score_dict['test']['macro'].append(float(json_data['predict_macro-f1']))
                 score_dict['params'].append(pooling_method)
        except:
           continue

    temp_stats = copy.deepcopy(score_dict)
    seed_scores = [(idx, score) for (idx, score) in enumerate(score_dict['dev']['micro'])]
    sorted_scores = sorted(seed_scores, key=lambda tup: tup[1])
    sorted_ids = [idx for idx, score in sorted_scores]
    temp_stats['params'] = [score_dict['params'][idx] for idx in sorted_ids]
    temp_stats['dev']['micro'] = [score_dict['dev']['micro'][idx] for idx in sorted_ids]
    temp_stats['dev']['macro'] = [score_dict['dev']['macro'][idx] for idx in sorted_ids]
    temp_stats['test']['micro'] = [score_dict['test']['micro'][idx] for idx in sorted_ids]
    temp_stats['test']['macro'] = [score_dict['test']['macro'][idx] for idx in sorted_ids]

    for idx, score in enumerate(temp_stats['dev']['micro']):
        report_line = f'POOLING: {temp_stats["params"][idx]:>10}\t'
        report_line += f'MICRO-F1: {temp_stats["dev"]["micro"][idx]*100:.1f}\t'
        report_line += f'MACRO-F1: {temp_stats["dev"]["macro"][idx]*100:.1f}\t'
        report_line += ' | '
        report_line += f'MICRO-F1: {temp_stats["test"]["micro"][idx]*100:.1f}\t'
        report_line += f'MACRO-F1: {temp_stats["test"]["macro"][idx]*100:.1f}\n'
        report_line += '-' * 200 + '\n'
        print(report_line)


if __name__ == '__main__':
    main()

