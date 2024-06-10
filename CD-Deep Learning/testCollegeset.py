
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
import torchvision.transforms as transforms
from network.CGNet import CGNet
from network.p2v import P2VNet
import torch.utils.data as data
import time

start=time.time()
class Test_Dataset(data.Dataset):
    def __init__(self, root, trainsize):
        self.trainsize = trainsize
        # get filenames
        image_root_A =  root + 'A/'
        image_root_B =  root + 'B/'
        
        self.images_A = [image_root_A + f for f in os.listdir(image_root_A) if f.endswith('.jpg') or f.endswith('.png')]
        self.images_B = [image_root_B + f for f in os.listdir(image_root_B) if f.endswith('.jpg') or f.endswith('.png')]
        self.images_A = sorted(self.images_A)
        self.images_B = sorted(self.images_B)
        self.filter_files()
        # transforms
        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])
        # get size of dataset
        self.size = len(self.images_A)

    def __getitem__(self, index):
        # read imgs/gts/grads/depths
        image_A = self.rgb_loader(self.images_A[index])
        image_B = self.rgb_loader(self.images_B[index])
        # data augumentation

        image_A = self.img_transform(image_A)
        image_B = self.img_transform(image_B)
        file_name = self.images_A[index].split('/')[-1][:-len(".png")]

        return image_A, image_B, file_name

    def filter_files(self):
        assert len(self.images_A) == len(self.images_B)
        images_A = []
        images_B = []
        for img_A_path, img_B_path in zip(self.images_A, self.images_B):
            img_A = Image.open(img_A_path)
            img_B = Image.open(img_B_path)
            
            if img_A.size == img_B.size:
                    images_A.append(img_A_path)
                    images_B.append(img_B_path)
                    

        self.images_A = images_A
        self.images_B = images_B
        

    def rgb_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('RGB')

    def binary_loader(self, path):
        with open(path, 'rb') as f:
            img = Image.open(f)
            return img.convert('L')

    def __len__(self):
        return self.size

def get_testset_loader(root, batchsize, trainsize, num_workers=1, shuffle=True, pin_memory=True):

    dataset =Test_Dataset(root = root, trainsize= trainsize)
    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batchsize,
                                  shuffle=shuffle,
                                  num_workers=num_workers,
                                  pin_memory=pin_memory)
    return data_loader




def test(test_loader, Eva_test, save_path, net):
    print("Start Validating!")


    net.train(False)
    net.eval()
    index=0
    for i, (A, B, filename) in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            A = A.cuda()
            B = B.cuda()
            preds = net(A,B)
            output = F.sigmoid(preds[1])
            output[output >= 0.5] = 1
            output[output < 0.5] = 0
            
            for i in range(output.shape[0]):
                probs_array = (torch.squeeze(output[i])).data.cpu().numpy()
                final_mask = probs_array * 255
                final_mask = final_mask.astype(np.uint8)
                print(filename[i])
                final_savepath = save_path + filename[i] + '.png'
                im = Image.fromarray(final_mask)
                im.save(final_savepath)



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


    opt.save_path = "./trainval/test_result/"
    test_loader = get_testset_loader(opt.test_root, opt.batchsize, opt.trainsize, num_workers=2, shuffle=False, pin_memory=True)
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