import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from dataset import DatasetFromHdf5
from torch.autograd import Variable
from scipy.misc import imsave
from scipy.ndimage import imread
from PIL import Image
import numpy as np
import time, math
import matplotlib.pyplot as plt
from KC_Pruning import KC_Pruning
from KC_vdsr import KC_Net



parser = argparse.ArgumentParser(description="PyTorch KC_Retrain")
parser.add_argument("--cuda", action="store_true", help="use cuda?")
parser.add_argument("--model", default="checkpoint/KC_model/KC_model_epoch_50.pth", type=str, help="model path")
parser.add_argument("--image", default="Set14/monarch", type=str, help="image name")
parser.add_argument("--scale", default=3, type=int, help="scale factor, Default: 4")
parser.add_argument("--gpus", default="0", type=str, help="gpu ids (default: 0)")
parser.add_argument("--batchSize", type=int, default=128)
parser.add_argument("--nEpochs", type=int, default=50)
parser.add_argument("--lr", type=float, default=0.1)
parser.add_argument("--step", type=int, default=10)
parser.add_argument("--resume", default="", type=str)
parser.add_argument("--start-epoch",default=1, type=int)
parser.add_argument("--threads", type=int, default=1)
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float)
parser.add_argument("--pretrained", default='', type=str)
parser.add_argument("--momentum", default=0.9, type=float)


def PSNR(pred, gt, shave_border=0):
    height, width = pred.shape[:2]
    pred = pred[shave_border:height - shave_border, shave_border:width - shave_border]
    gt = gt[shave_border:height - shave_border, shave_border:width - shave_border]

    imdff = pred - gt
    rmse = math.sqrt(np.mean(imdff ** 2))
    if rmse == 0:
        return 100
    return 20 * math.log10(255.0 / rmse)

def colorize(y, ycbcr):
    img = np.zeros((y.shape[0], y.shape[1], 3), np.uint8)
    img[:,:,0] = y
    img[:,:,1] = ycbcr[:,:,1]
    img[:,:,2] = ycbcr[:,:,2]
    img = Image.fromarray(img, "YCbCr").convert("RGB")
    return img


opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")


new_model = torch.load(opt.model, map_location=lambda storage, loc: storage)["new_model"]

im_gt_ycbcr = imread(opt.image + ".bmp", mode="YCbCr")
im_b_ycbcr = imread(opt.image + "_scale_"+ str(opt.scale) + ".bmp", mode="YCbCr")
im_gt_y = im_gt_ycbcr[:,:,0].astype(float)
im_b_y = im_b_ycbcr[:,:,0].astype(float)




psnr_bicubic = PSNR(im_gt_y, im_b_y,shave_border=opt.scale)

im_input = im_b_y/255.

im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])


if cuda:
    new_model = new_model.cuda()
    im_input = im_input.cuda()
else:
    new_model = new_model.cpu()

start_time = time.time()
out = new_model(im_input)
elapsed_time = time.time() - start_time



out = out.cpu( )


temp=list(new_model.parameters())
n_w_sum=0
w_sum=64*64*18+64*2

for weights in temp:
    print(weights.shape)
    n_w_sum+=weights.shape[0]*weights.shape[1]
    
print(float(n_w_sum*100.0)/float(w_sum))






im_h_y = out.data[0].numpy().astype(np.float32)

im_h_y = im_h_y * 255.
im_h_y[im_h_y<0] = 0
im_h_y[im_h_y>255.] = 255.

psnr_predicted = PSNR(im_gt_y, im_h_y[0,:,:], shave_border=opt.scale)

im_h = colorize(im_h_y[0,:,:], im_b_ycbcr)
im_gt = Image.fromarray(im_gt_ycbcr,"YCbCr").convert("RGB")
im_b = Image.fromarray(im_b_ycbcr, "YCbCr").convert("RGB")
print("Scale=",opt.scale)
print("PSNR_predicted=", psnr_predicted)
print("PSNR_bicubic=", psnr_bicubic)
print("It takes {}s for processing".format(elapsed_time))
imsave('result/KC_SR.png',im_h)
imsave('result/LR.png',im_b)
imsave('result/HR.png',im_gt)
fig = plt.figure()
ax = plt.subplot("131")
ax.imshow(im_gt)
ax.set_title("GT")
ax = plt.subplot("132")
ax.imshow(im_b)
ax.set_title("Input(bicubic)")
ax = plt.subplot("133")
ax.imshow(im_h)
ax.set_title("Output(vdsr)")
plt.show()

