'''
Concrete SettingModule class for a specific experimental SettingModule
'''

# Copyright (c) 2017-Current Jiawei Zhang <jiawei@ifmlab.org>
# License: TBD

from code.base_class.setting import setting
from sklearn.model_selection import train_test_split
import numpy as np


class Setting_Train_Test_Split(setting):
    fold = 1

    def load_run_save_evaluate(self):

        loaded_data = self.dataset.load()

        self.method.data = {
            'train': loaded_data['train'],
            'test': loaded_data['test']
        }
        learned_result = self.method.run()

        if self.result is not None:
            self.result.data = learned_result
            self.result.save()

        # evaluate
        self.evaluate.data = learned_result

        return self.evaluate.evaluate(), None