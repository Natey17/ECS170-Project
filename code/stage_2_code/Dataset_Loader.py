from code.base_class.dataset import dataset
class Dataset_Loader(dataset):
    data = None
    dataset_source_folder_path = None
    dataset_source_file_name = None  # should be [train_file, test_file]

    def __init__(self, dName=None, dDescription=None):
        super().__init__(dName, dDescription)

    def load_file(self, file_path):
        X = []
        y = []

        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip('\n')
                elements = [int(i) for i in line.split(',')]

                y.append(elements[0])        # label is FIRST
                X.append(elements[1:])      # features are rest (784)

        return X, y

    def load(self):
        print('loading data...')

        # Expecting [train.csv, test.csv]
        train_file = self.dataset_source_folder_path + self.dataset_source_file_name[0]
        test_file = self.dataset_source_folder_path + self.dataset_source_file_name[1]

        X_train, y_train = self.load_file(train_file)
        X_test, y_test = self.load_file(test_file)

        return {
            'train': {'X': X_train, 'y': y_train},
            'test': {'X': X_test, 'y': y_test}
        }