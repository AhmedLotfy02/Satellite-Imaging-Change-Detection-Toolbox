
import time
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from utils import data_loader
from tqdm import tqdm
from utils.metrics import Evaluator
from PIL import Image

from network.CGNet import CGNet
from network.p2v import P2VNet

import time
start=time.time()
def test(test_loader, Eva_test, save_path, net):
    print("Start Validating!")


    net.train(False)
    net.eval()
    for i, (A, B, mask, filename) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            Y = mask.cuda()
            preds = net(A,B)
            output = F.sigmoid(preds[1])
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            pred = output.data.cpu().numpy().astype(int)
            target = Y.cpu().numpy()

            for i in range(output.shape[0]):
                probs_array = (torch.squeeze(output[i])).data.cpu().numpy()
                final_mask = probs_array * 255
                final_mask = final_mask.astype(np.uint8)
                final_savepath = save_path + filename[i] + '.png'
                im = Image.fromarray(final_mask)
                im.save(final_savepath)

            Eva_test.add_batch(target, pred)

    IoU = Eva_test.Intersection_over_Union()
    Pre = Eva_test.Precision()
    Recall = Eva_test.Recall()
    F1 = Eva_test.F1()
    OA=Eva_test.OA()
    print('[Test] F1: %.4f, Precision:%.4f, Recall: %.4f, OA: %.4f, IoU: %.4f' % ( F1[1],Pre[1],Recall[1],OA[1],IoU[1]))
    print('F1-Score: Precision: Recall: OA: IoU: ')
    print('{:.2f}\{:.2f}\{:.2f}\{:.2f}\{:.2f}'.format(F1[1] * 100, Pre[1] * 100, Recall[1] * 100, OA[1] * 100,IoU[1] * 100))
    print('{:.2f} {:.2f} {:.2f} {:.2f} {:.2f}\n'.format(F1[1] * 100, Pre[1] * 100, Recall[1] * 100, OA[1] * 100,IoU[1] * 100))


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--trainsize', type=int, default=256, help='training dataset size')
    parser.add_argument('--gpu_id', type=str, default='0', help='train use gpu')  
    parser.add_argument('--data_name', type=str, default='trainProject', 
                        help='the test rgb images root')
    parser.add_argument('--model_name', type=str, default='CGNet',
                        help='the test rgb images root')
    parser.add_argument('--save_path', type=str,
                        default='./test_result/')
    opt = parser.parse_args()

    # set the device for training
    if opt.gpu_id == '0':
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"
        print('USE GPU 0')
   
    opt.test_root='./trainval/test/'


    opt.save_path = opt.save_path + opt.data_name + '/' + opt.model_name + '/'
    test_loader = data_loader.get_test_loader(opt.test_root, opt.batchsize, opt.trainsize, num_workers=2, shuffle=False, pin_memory=True)
    Eva_test = Evaluator(num_class=2)

    if opt.model_name == 'CGNet':
        model = CGNet().cuda()
    elif opt.model_name == 'p2v':
        model=P2VNet().cuda()

    opt.load = './output/' + opt.data_name + '/' + opt.model_name + '_best_iou.pth'
    
    if opt.load is not None:
        model.load_state_dict(torch.load(opt.load))
        print('load model from ', opt.load)


    save_path = opt.save_path
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    test(test_loader, Eva_test, opt.save_path, model)

end=time.time()
print('Test Time :',end-start)