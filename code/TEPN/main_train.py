import os
import torch
import torch.optim as optim
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import r2_score

# Modules
from model.TFP import TFP_module as TEP
from model.TEE import TEE_module as TEE
from model.reprojection import reprojection

# Dataset imports
from dataset import (
    VideoFrameDataset,
    create_dataset,
    create_dataset_v
)

# Utils
from utils.logger import Logger
from utils.loss import (
    L1_Charbonnier_loss,
    reprojection_loss
)
from para import Parameter

# GPU setting
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

if __name__ == '__main__':
    # Training setup - CUDA optimizations
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    # Initialize parameters and datasets
    para = Parameter().args
    dataset = VideoFrameDataset(para, flag=0)
    dataset_valid = VideoFrameDataset(para, flag=1)

    # Setup logger and tensorboard
    logger = Logger(para)
    logger.writer = SummaryWriter(logger.save_dir)
    logger('building {} model ...'.format(para.model), prefix='\n')

    # Create dataloaders
    dataloader = create_dataset(dataset, para)
    dataloader_valid = create_dataset_v(dataset_valid, para)

    # Initialize models
    TEP_ = TEP(para)
    TEE_ = TEE(para)

    # Setup device and move models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    TEP_ = TEP_.to(device)
    TEE_ = TEE_.to(device)

    # Log model structures
    logger('TEP_structure:', TEP_, verbose=False)
    logger('TEE_structure:', TEE_, verbose=False)

    # Initialize loss functions
    get_rec_loss = L1_Charbonnier_loss()
    get_cn2_l1_loss = L1_Charbonnier_loss()
    get_con_loss = reprojection_loss()

    # Initialize training variables
    total_iters = 0
    epoch_r2_best = 0
    epoch_loss_best = 1e10

    # Setup optimizers
    optimizer = optim.Adam(TEE_.parameters(), lr=para.lr, betas=[0.9, 0.999])
    optimizer_cn2 = optim.Adam(TEP_.parameters(), lr=para.lr * 2, betas=[0.9, 0.999])

    # resume from a checkpoint
    if para.resume:
        if os.path.isfile(para.resume_file):
            checkpoint = torch.load(para.resume_file, map_location=lambda storage, loc: storage.cuda(0))
            logger('loading checkpoint {} ...'.format(para.resume_file))
            logger.register_dict = checkpoint['register_dict']
            para.start_epoch = checkpoint['epoch'] + 1
            TEP_.load_state_dict(checkpoint['state_dict'], strict=False)
            deblur_model_dict = checkpoint['state_dict_deblur']
            TEE_.load_state_dict(deblur_model_dict)
            optimizer.load_state_dict(checkpoint['optimizer'])
            optimizer_cn2.load_state_dict(checkpoint['optimizer_cn2'])
        else:
            msg = 'no check point found at {}'.format(para.resume_file)
            logger(msg, verbose=False)
            raise FileNotFoundError(msg)
    len_traindata = len(dataloader)
    ratio_len = int(1 * len_traindata)
    len_testdata = len(dataset_valid)

    # Training loop
    for epoch in range(para.start_epoch, para.n_epochs + 1):
        epoch_loss = 0
        epoch_loss_cn2 = 0
        total_iters = 0
        epoch_r2 = 0
        t = tqdm(dataloader, desc=f"Epoch {epoch}/{para.n_epochs}", ncols=250)
        for i, data in enumerate(t):
            total_iters += para.batch_size
            input = data['input'].to(device)
            input = torch.squeeze(input)
            gt = data['gt'].to(device)
            gt = gt.squeeze().unsqueeze(1)
            label_cn2 = data['label_cn2'].to(device)
            label_integral = data['label_integral'].to(device)
            output = TEE_(input)

            # get the effect of turbulence
            tur_eff = input - output.expand(5, 9, 3, 320, 480)
            cn2_integral_out, cn2_out = TEP_(tur_eff)
            cn2_mat = torch.cat([cn2_integral_out,cn2_out],dim=1)
            cn2_mat_gt = torch.cat([label_integral,label_cn2],dim=1)
            cn2_mat_reproject = reprojection(cn2_mat)

            # calculate the reprojection loss
            loss_rpj = 5 * get_con_loss(cn2_mat_reproject, cn2_mat_gt)

            # calculate the reconstruction loss
            loss_rec = 50 * get_rec_loss(output, gt)

            # calculate the turbulence loss
            loss_cn2_2d = get_cn2_l1_loss(label_integral, cn2_integral_out)
            loss_cn2_3d = get_cn2_l1_loss(label_cn2, cn2_out)
            loss_turb = 30 * loss_cn2_2d + 10 * loss_cn2_3d

            # calculate the total loss
            loss_all = loss_rec + loss_turb + loss_rpj
            # update the parameters
            optimizer.zero_grad()
            optimizer_cn2.zero_grad()
            loss_all.backward()
            clip_grad_norm_(TEE_.parameters(), max_norm=20, norm_type=2)
            clip_grad_norm_(TEP_.parameters(), max_norm=20, norm_type=2)
            optimizer_cn2.step()
            optimizer.step()

            # calculate the p and r2 between label_cn2 and cn2_out
            p = torch.corrcoef(torch.concat([label_cn2.view((1,-1)),cn2_out.view((1,-1))],dim=0))[1,0]
            r2 = r2_score(label_cn2.view(-1).cpu().detach().numpy(),cn2_out.view(-1).cpu().detach().numpy())
            epoch_loss += loss_all.item()
            epoch_loss_cn2 += loss_turb.item()
            epoch_r2 += r2.item()
            message = '(epoch: {}, iter: {}, loss: {},loss_cn2:{}, loss_cn2_3d:{} , r2:{})'.format(
                epoch, total_iters, loss_rec, loss_turb, loss_cn2_3d, r2)
            t.set_description(message)
            t.refresh()
        # print the training information
        epoch_loss = epoch_loss / (ratio_len / para.batch_size)
        epoch_loss_cn2 = epoch_loss_cn2 / (ratio_len / para.batch_size)
        epoch_r2 = epoch_r2/ (ratio_len / para.batch_size)
        print('Epoch {}: loss of the Rnn is {}\n loss_cn2 of the Run is {}\n r2 of the Run is {}\n'.format(epoch, epoch_loss,epoch_loss_cn2 ,epoch_r2))
        with open(os.path.join(para.savepath, 'Epoch_loss_logger_{}.txt'.format(para.train_time)), 'a') as log_file:
            log_file.write(
                'Epoch : {},  loss :{}, loss_cn2_3d:{}, lr:{}, lr_cn2:{}, p:{}\n'.format(epoch, epoch_loss,
                epoch_loss_cn2,optimizer.param_groups[0]['lr'],
                optimizer_cn2.param_groups[0]['lr'],epoch_r2))
        checkpoint = {
            'epoch': epoch,
            'model': para.model,
            'state_dict_tep': TEP_.state_dict(),
            'state_dict_tee': TEE_.state_dict(),
            'register_dict': logger.register_dict,
            'optimizer': optimizer.state_dict(),
            'optimizer_cn2': optimizer_cn2.state_dict(),
        }
        with torch.no_grad():
            test_epoch_loss = 0
            test_epoch_loss_cn2=0
            valid_epoch_r2 = 0
            valid_epoch_p = 0
            valid_epoch_mae = 0
            valid_epoch_mse = 0
            valid_rmse = 0
            total_iters=0
            t_valid = tqdm(dataloader_valid, desc=f"Testing Epoch {epoch}/{para.n_epochs}", ncols=200)
            for i, data in enumerate(t_valid):
                total_iters += para.batch_size
                input = data['input'].to(device)
                input = torch.squeeze(input)
                gt = data['gt'].to(device)
                gt = gt.squeeze().unsqueeze(1)
                label_cn2 = data['label_cn2'].to(device)
                label_integral = data['label_integral'].to(device)
                output = TEE_(input)
                tur_eff = input - output.expand(5, 9, 3, 320, 480)
                cn2_integral_out, cn2_out = TEP_(tur_eff)
                p = torch.corrcoef(torch.concat([label_cn2.view(1,-1),cn2_out.reshape(1,-1)],dim = 0))[1,0]
                r2 = r2_score(label_cn2.view(-1).cpu().detach().numpy(),cn2_out.reshape(-1).cpu().detach().numpy())
                mae = torch.mean(torch.abs(label_cn2 - cn2_out))
                mse = torch.mean((label_cn2 - cn2_out) ** 2)
                rmse = torch.mean((label_cn2 - cn2_out) ** 2).sqrt()
                valid_epoch_p += p.item()
                valid_epoch_mae += mae.item()
                valid_epoch_mse += mse.item()
                valid_rmse += rmse.item()
                valid_epoch_r2 += r2.item()
                message_ = '(test_epoch:{}, iter:{}, mae:{}, mse:{}, p:{},r2:{})'.format(epoch, total_iters,mae, mse,p,r2)
                t_valid.write(message_)
                t_valid.update()
                with open(para.log_path, 'a') as log_file:
                    log_file.write('{}\n'.format(message_))
            valid_rmse = valid_rmse / (len(dataset_valid) / para.batch_size)
            valid_epoch_p = valid_epoch_p / (len(dataset_valid) / para.batch_size)
            valid_epoch_mae = valid_epoch_mae / (len(dataset_valid) / para.batch_size)
            valid_epoch_mse = valid_epoch_mse / (len(dataset_valid) / para.batch_size)
            valid_epoch_r2 = valid_epoch_r2 / (len(dataset_valid) / para.batch_size)
            print('mae is {}, mse is {}, rmse is {}, r2 is {}, p is {}'.format(valid_epoch_mae, valid_epoch_mse, valid_rmse, valid_epoch_r2, valid_epoch_p))
            with open(os.path.join(para.savepath, 'Epoch_loss_logger_{}.txt'.format(para.train_time)), 'a') as log_file:
                log_file.write(
                    'valid___Epoch : mae is {}, mse is {}, rmse is {}, r2 is {}, p is {}'.format(valid_epoch_mae, valid_epoch_mse, valid_rmse, valid_epoch_r2, valid_epoch_p))

        if epoch == 1:
            logger.save_best(checkpoint)
            epoch_r2_best = valid_epoch_r2
        
        elif valid_epoch_r2 > epoch_r2_best:
            epoch_r2_best = valid_epoch_r2
            logger.save_best(checkpoint)
        elif epoch % 5==0:
            logger.save(checkpoint, epoch)

        torch.cuda.empty_cache()
