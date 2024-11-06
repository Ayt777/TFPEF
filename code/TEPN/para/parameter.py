import argparse
import time

class Parameter(object):
    def __init__(self):
        self.args = self.extract_args()

    def extract_args(self):
        self.parser = argparse.ArgumentParser(description='Video extract Cn2')
        self.parser.add_argument('--train_time', type=str, default=time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime()))
        self.parser.add_argument('--data_root', type=str, default='./MATID/train')
        self.parser.add_argument('--data_test_root', type=str, default='./MATID/test')
        self.parser.add_argument('--time_length', type=int, default=9, help='the num of frames')

        # tfp parameters
        self.parser.add_argument('--n_features', type=int, default=8, help='base # of channels for Conv')
        self.parser.add_argument('--n_blocks', type=int, default=8, help='# of blocks in middle part of the model')
        self.parser.add_argument('--activation', type=str, default='leakyrelu', help='activation function')
        # tee model
        self.parser.add_argument('--n_features_tee', type=int, default=8, help='base # of channels for Conv')
        self.parser.add_argument('--n_blocks_tee', type=int, default=8, help='# of blocks in middle part of the model')
        self.parser.add_argument('--future_frames', type=int, default=4, help='use # of future frames')
        self.parser.add_argument('--past_frames', type=int, default=4, help='use # of past frames')
        self.parser.add_argument('--activation_tee', type=str, default='gelu', help='activation function')
        self.parser.add_argument('--n_frames', type=int, default=9, help='use # of  frames')
        self.parser.add_argument('--data_flag', type=int, default=0)
        #train parametersparameter.py
        self.parser.add_argument('--resume', type=bool, default=False)
        self.parser.add_argument('--resume_file', type=str,default='')
        self.parser.add_argument('--lr', type=float, default=1e-4,help='learning rate')
        self.parser.add_argument('--start_epoch', type=int, default=1)
        self.parser.add_argument('--n_epochs', type=int, default=100)
        self.parser.add_argument('--batch_size', type=int, default=1)
        self.parser.add_argument('--shuffle', type=bool, default=True)
        self.parser.add_argument('--print_freq', type=int, default=3)
        self.parser.add_argument('--savepath', type=str, default='./experiment')
        self.parser.add_argument('--model', type=str, default='TEPN')
        # pretrained_reprojection_model
        self.parser.add_argument('--pretrained_reprojection_file', type=str, default='./experiment/2023_08_14_21_04_01_rdb_cn2_only_cn2/checkpoint_500.pth.tar')
        self.parser.add_argument('--dataset', type=str, default='MATID' )
        self.parser.add_argument('--log_path', type=str, default='./experiment/log.txt')
        args, _ = self.parser.parse_known_args()

        return args
