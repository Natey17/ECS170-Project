'''
Multi-Layer Perceptron script for Stage 2
'''

from code.stage_2_code.Result_Saver import Result_Saver
from code.stage_2_code.Dataset_Loader import Dataset_Loader
from code.stage_2_code.Method_MLP import Method_MLP
from code.stage_2_code.Setting_Train_Test_Split import Setting_Train_Test_Split
from code.stage_2_code.Evaluate_Accuracy import Evaluate_Accuracy
import numpy as np
import torch
import matplotlib.pyplot as plt

# ---- Multi-Layer Perceptron script ----
if 1:
    # ---- parameter section -------------------------------
    np.random.seed(2)
    torch.manual_seed(2)
    # ------------------------------------------------------

    # ---- object initialization section -------------------
    data_obj = Dataset_Loader('mnist', '')
    data_obj.dataset_source_folder_path = '../../data/stage_2_data/'
    data_obj.dataset_source_file_name = ['train.csv', 'test.csv']

    method_obj = Method_MLP('multi-layer perceptron', '')

    result_obj = Result_Saver('saver', '')
    result_obj.result_destination_folder_path = '../../result/stage_2_result/MLP_'
    result_obj.result_destination_file_name = 'prediction_result'

    setting_obj = Setting_Train_Test_Split('train test split', '')

    evaluate_obj = Evaluate_Accuracy('accuracy', '')
    # ------------------------------------------------------

    # ---- running section ---------------------------------
    print('************ Start ************')

    setting_obj.prepare(data_obj, method_obj, result_obj, evaluate_obj)
    setting_obj.print_setup_summary()

    mean_score, std_score = setting_obj.load_run_save_evaluate()

    print('************ Overall Performance ************')
    print('MLP Accuracy: ' + str(mean_score))
    print('************ Finish ************')

    # ---- learning curve plot ----------------------------
    plt.figure(figsize=(10, 6))
    plt.plot(range(method_obj.max_epoch), method_obj.loss_history,
             color='blue', linewidth=2)
    plt.title('MLP Learning Curve - MNIST')
    plt.xlabel('Epoch')
    plt.ylabel('Training Loss')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('../../result/stage_2_result/learning_curve.png')
    plt.show()
    print('Learning curve saved!')