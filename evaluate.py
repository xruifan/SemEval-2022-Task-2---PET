"""
Author: Xuan-Rui Fan
Email: serfinxx@gmail.com
Date: 07 Apr 2022
"""

from pet import InputExample
from pet.wrapper import TransformerModelWrapper

import csv
import os
from sklearn.metrics import f1_score

root_model_dir = '.\semeval\\'

eval_data_dict = {'Eval': '.\semeval_data\\final_test.csv', 
        'Test': '.\semeval_data\\test\\final_test.csv'}

for eval_data, eval_data_path in eval_data_dict.items():
    # read test data
    data = []
    labels = []
    with open(eval_data_path, encoding='utf8') as f:
                reader = csv.DictReader(f, delimiter='\t')
                for idx, row in enumerate(reader):
                    sentence = row['Sentence']
                    label = row['Label']

                    example = InputExample(guid=0, text_a=sentence)
                    data.append(example)
                    labels.append(label)

    for subdir in os.listdir(root_model_dir):

        print("Evaluating {} on {}......".format(subdir, eval_data))
        task, num, pattern = subdir.split('-')
        
        model_path = os.path.join(root_model_dir, subdir, 'final\p0-i0\\')

        # load model
        model = TransformerModelWrapper.from_pretrained(model_path)

        # move model to cuda
        model.model.to('cuda')

        pred_logits = model.eval(data, device='cuda', per_gpu_eval_batch_size=1)['logits']

        # get label indexes for classes
        label_idx_idiom = model.config.label_list.index('0')
        label_idx_non_idiom = model.config.label_list.index('1')
        pred_labels = []
        for logits in pred_logits:
            if logits[label_idx_idiom] > logits[label_idx_non_idiom]:
                pred_labels.append('0')
            else: 
                pred_labels.append('1')

        assert len(labels) == len(pred_labels)

        # calculate f1 score
        f1_macro_score = f1_score(labels, pred_labels, average='macro')

        # check if path exists
        output_path = os.path.join('.\\', eval_data, task + '-' + pattern, num)
        if not os.path.exists(output_path):
            os.makedirs(output_path)

        # write result
        head = ['label', 'prediction']
        with open(os.path.join(output_path, '{}_predictions.csv'.format(eval_data)), 'w', encoding='utf-8', newline='') as f: 
                write = csv.writer(f, delimiter='\t')
                write.writerow(head)
                for idx, label in enumerate(labels):
                    write.writerow([label, pred_labels[idx]])

        with open(os.path.join(output_path, 'f1_macro_on_{}.txt'.format(eval_data)), 'w') as f:
            f.write('final f1-macro score on {}: {}'.format(eval_data, f1_macro_score))
    