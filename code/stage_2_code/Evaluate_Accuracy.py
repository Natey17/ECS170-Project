'''
Concrete Evaluate class for a specific evaluation metrics
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.evaluate import evaluate
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score


class Evaluate_Accuracy(evaluate):
    data = None

    def evaluate(self):
        print('evaluating performance...')

        true_y = self.data['true_y']
        pred_y = self.data['pred_y']

        accuracy = accuracy_score(true_y, pred_y)
        f1 = f1_score(true_y, pred_y, average='weighted')
        recall = recall_score(true_y, pred_y, average='weighted')
        precision = precision_score(true_y, pred_y, average='weighted', zero_division=0)

        print('Accuracy:', accuracy)
        print('F1 (weighted):', f1)
        print('Recall (weighted):', recall)
        print('Precision (weighted):', precision)

        return accuracy