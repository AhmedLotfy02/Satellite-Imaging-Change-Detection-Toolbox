import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch import optim
import utils.visualization as visual
from utils import data_loader
from torch.optim import lr_scheduler
from tqdm import tqdm
import random
from utils.metrics import Evaluator

from network.CGNet import CGNet
from network.p2v import P2VNet
import time
start=time.time()

def seed_everything(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def train(train_loader, val_loader, Eva_train, Eva_val, data_name, save_path, net, criterion, optimizer, num_epoches):
    vis = visual.Visualization()
    vis.create_summary(data_name)
    global best_iou
    epoch_loss = 0
    net.train(True)

    length = 0
    # I used tqdm to make a progress bar
    for i, (A, B, mask) in enumerate(tqdm(train_loader)):
        A = A.cuda()
        B = B.cuda()
        Y = mask.cuda()
        optimizer.zero_grad()
        preds = net(A,B)
        loss = criterion(preds[0], Y)  + criterion(preds[1], Y)
        # ---- loss function ----
        loss.backward()
        optimizer.step()
        # scheduler.step()
        epoch_loss += loss.item()

        output = F.sigmoid(preds[1])
        output[output >= 0.5] = 1
        output[output < 0.5] = 0
        pred = output.data.cpu().numpy().astype(int)
        target = Y.cpu().numpy().astype(int)
        
        Eva_train.add_batch(target, pred)
        # print("Length of pred:",len(pred))
        for k in range(len(pred)):
            IoU_img = Eva_train.Calculate_IoU(pred[k], target[k])
            Eva_train.mIoU+=IoU_img
  

        length += 1
    IoU = Eva_train.Intersection_over_Union()[1]
    
    print("DEBUGING mIoU/12:",Eva_train.mIoU/(12*length))
    print("DEBUGING mIoU:",Eva_train.mIoU)
    print("Number of batches:",length)
    
    Pre = Eva_train.Precision()[1]
    Recall = Eva_train.Recall()[1]
    F1 = Eva_train.F1()[1]
    train_loss = epoch_loss / length

    vis.add_scalar(epoch, IoU, 'mIoU')
    vis.add_scalar(epoch, Pre, 'Precision')
    vis.add_scalar(epoch, Recall, 'Recall')
    vis.add_scalar(epoch, F1, 'F1')
    vis.add_scalar(epoch, train_loss, 'train_loss')

    print(
        'Epoch [%d/%d], Loss: %.4f,\n[Training]IoU: %.4f,[Training]mIoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (
            epoch, num_epoches, \
            train_loss, \
            IoU,Eva_train.mIoU/(12*length), Pre, Recall, F1))
    

    print("Start Validating....")
    length = 0
    net.train(False)
    net.eval()
    for i, (A, B, mask, filename) in enumerate(tqdm(val_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            preds = net(A,B)[1]
            output = F.sigmoid(preds)
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy().astype(int)

            Eva_val.add_batch(target, pred)
            for k in range(len(pred)):
                img_IoU = Eva_val.Calculate_IoU(pred[k], target[k])
                Eva_val.mIoU+=img_IoU
    
            length += 1
    print("Number of batches:",length)
    IoU = Eva_val.Intersection_over_Union()
    newmIoU = Eva_val.mIoU/(12*length)
    Pre = Eva_val.Precision()
    Recall = Eva_val.Recall()
    F1 = Eva_val.F1()

    print('[Validation] IoU: %.4f,mIoU: %.4f, Precision:%.4f, Recall: %.4f, F1: %.4f' % (IoU[1],newmIoU ,Pre[1], Recall[1], F1[1]))
    new_iou = IoU[1]
    if newmIoU >= best_iou:
        best_iou = newmIoU
        best_epoch = epoch
        best_net = net.state_dict()
        print('Best Model Iou :%.4f, mIoU:%.4f; F1 :%.4f; Best epoch : %d' % (IoU[1],best_iou, F1[1], best_epoch))
        torch.save(best_net, save_path + '_best_iou.pth')
    # Save the model for each epoch 
    torch.save(net.state_dict(), save_path + '_epoch_%d.pth' % epoch)
    print('Best Model Iou :%.4f; F1 :%.4f' % (best_iou, F1[1]))
    vis.close_summary()


if __name__ == '__main__':
    seed_everything(42)
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--epoch', type=int, default=50, help='epoch number') 
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--batchsize', type=int, default=8, help='training batch size') 
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--gpu_id', type=str, default='2', help='train use gpu')  
    parser.add_argument('--data_name', type=str, default='trainProject', 
                        help='the test rgb images root')
    parser.add_argument('--model_name', type=str, default='CGNet',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./output/')
    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
   
    opt.save_path = opt.save_path + opt.data_name + '/' + opt.model_name
    
    if opt.data_name == 'trainProject':
        opt.train_root = './trainval/train/'
        opt.val_root = './trainval/val/'
    else:
        opt.train_root='./trainval/train/'
        opt.val_root='./trainval/val/'
    

    train_loader = data_loader.get_loader(opt.train_root, opt.batchsize, opt.trainsize, num_workers=2, shuffle=True, pin_memory=True)
    val_loader = data_loader.get_test_loader(opt.val_root, opt.batchsize, opt.trainsize, num_workers=2, shuffle=False, pin_memory=True)
    Eva_train = Evaluator(num_class = 2)
    Eva_val = Evaluator(num_class=2)

    if opt.model_name == 'CGNet':
        model = CGNet().cuda()
    elif opt.model_name == 'p2v':
        model=P2VNet(in_ch=3).cuda()
    # We will use binary cross entropy as our problem is binary classifcation associated with sigmoid activation 
    criterion = nn.BCEWithLogitsLoss().cuda()
    # weight_decay is L2 regularization to prevent overfitting  
    optimizer = torch.optim.AdamW(model.parameters(), lr=opt.lr, weight_decay=0.0025)
    # the learning rate is reset to its value after each 15 epochs
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=15, T_mult=2)

    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    data_name = opt.data_name
    best_iou = 0.0

    print("Start train...")
    for epoch in range(1, opt.epoch):
        for param_group in optimizer.param_groups:
            print(param_group['lr'])
        Eva_train.reset()
        Eva_val.reset()

        train(train_loader, val_loader, Eva_train, Eva_val, data_name, save_path, model, criterion, optimizer, opt.epoch)
        #move the learning rate scheduler one step
        lr_scheduler.step()

end=time.time()
print("FINISIHED")
print('Training Time :',end-start)