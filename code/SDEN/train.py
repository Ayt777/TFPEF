import torch
from torch.utils.data import Dataset
import pickle
from openstl.api import BaseExperiment
from openstl.utils import create_parser, default_parser
import os
import numpy as np
os.environ['CUBLAS_WORKSPACE_CONFIG']=':4096:8'

class CustomDataset(Dataset):
    def __init__(self, X, Y, mean=None, std=None, min=None, max=None, normalize=True, type='train', data_name='costom'):
        super(CustomDataset, self).__init__()
        self.X = X
        self.Y = Y
        self.mean = mean
        self.std = std
        self.min = min
        self.max = max
        self.type = type
        self.data_name = data_name

        if normalize:
            # get the mean/std values along the channel dimension
            if self.type == 'train':
                self.mean = self.X.mean(axis=(0, 1, 3, 4)).reshape(1, 1, -1, 1, 1)
                self.std = self.X.std(axis=(0, 1, 3, 4)).reshape(1, 1, -1, 1, 1)
                self.min = self.X.min(axis=(0, 1, 3, 4)).reshape(1, 1, -1, 1, 1)
                self.max = self.X.max(axis=(0, 1, 3, 4)).reshape(1, 1, -1, 1, 1)
            self.X = (self.X - self.mean) / self.std
            self.Y = (self.Y - self.mean) / self.std


    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        data = torch.tensor(self.X[index]).float()
        labels = torch.tensor(self.Y[index]).float()
        return data, labels
    
if __name__=="__main__":
    pre_seq_length = 4
    aft_seq_length = 4
    batch_size = 4
    # SimVP+gSTA  0717a 4->4
    custom_training_config = {
        'pre_seq_length': pre_seq_length,
        'aft_seq_length': aft_seq_length,
        'total_length': pre_seq_length + aft_seq_length,
        'batch_size': batch_size,
        'val_batch_size': batch_size,
        'epoch': 9,
        'lr': 0.003,

        'metrics': ['mse', 'mae','rmse'],

        'ex_name': 'custom_exp',
        'dataname': 'custom',
        'in_shape': [4, 20, 30, 30],
    }
    # SimVP+gSTA  0717a 4->4
    custom_model_config = {
        # For MetaVP models, the most important hyperparameters are: 
        # N_S, N_T, hid_S, hid_T, model_type
        'method': 'SimVP',
        # Users can either using a config file or directly set these hyperparameters 
        # 'config_file': 'configs/custom/example_model.py',
        
        # Here, we directly set these parameters
        'model_type': 'gSTA',
        # 4->4
        # 'N_S': 4,
        # 'N_T': 4,
        # 'hid_S': 64,
        # 'hid_T': 60,
        # 4->16
        'N_S':4,
        'N_T': 4,
        'hid_S': 64,
        'hid_T': 60,
    }
  
    # load the dataset
    with open('dataset_all_4_4.pkl', 'rb') as f:
        dataset = pickle.load(f)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = dataset['X_train'], dataset[
    'X_val'], dataset['X_test'], dataset['Y_train'], dataset['Y_val'], dataset['Y_test']
  
    train_set = CustomDataset(X=X_train, Y=Y_train, type='train')
    val_set = CustomDataset(X=X_val, Y=Y_val, mean=train_set.mean, std=train_set.std, min=train_set.min,max=train_set.max, type='val')
    test_set = CustomDataset(X=X_test, Y=Y_test, mean=train_set.mean, std=train_set.std, min=train_set.min,max=train_set.max, type='val')

    dataloader_train = torch.utils.data.DataLoader(
    train_set, batch_size=batch_size, shuffle=True, pin_memory=True,drop_last=False)
    dataloader_val = torch.utils.data.DataLoader(
    val_set, batch_size=batch_size, shuffle=False, pin_memory=True,drop_last=False)
    dataloader_test = torch.utils.data.DataLoader(
    test_set, batch_size=batch_size, shuffle=False, pin_memory=True,drop_last=False)


    args = create_parser().parse_args([])
    config = args.__dict__

    # update the training config
    config.update(custom_training_config)
    # update the model config
    config.update(custom_model_config)
    # fulfill with default values
    default_values = default_parser()
    for attribute in default_values.keys():
        if config[attribute] is None:
            config[attribute] = default_values[attribute]

    exp = BaseExperiment(args, dataloaders=(dataloader_train, dataloader_val, dataloader_test), strategy='auto')


    # methods/simvp.py  ==> models/simvp_model.py ==> modules/simvp_modules.py
    print('>'*35 + ' training ' + '<'*35)
    exp.train()

    print('>'*35 + ' testing  ' + '<'*35)
    exp.test()