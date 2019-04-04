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
parser.add_argument("--model", default="model/model_epoch_50.pth", type=str, help="model path")
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

def adjust_learning_rate(optimizer,epoch):
    lr = opt.lr *(0.1**(epoch//opt.step))
    return lr


def train(retraining_data_loader, optimizer, new_model, criterion, epoch):
    lr = adjust_learning_rate(optimizer, epoch-1)
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    print("Epoch = {}, lr = {}".format(epoch,optimizer.param_groups[0]["lr"]))
    new_model.train()
    for iteration, batch in enumerate(retraining_data_loader,1):
        input, target = Variable(batch[0]), Variable(batch[1],requires_grad=False)
        if opt.cuda:
            input = input.cuda()
            target = target.cuda()
        loss = criterion(new_model(input), target)
        optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm(new_model.parameters(),0.4)
        optimizer.step()
        if iteration%100 == 0:
            print("===> Epoch[{}] ({}/{}) : Loss:{:.10f}".format(epoch,iteration,len(retraining_data_loader),loss.data))


def save_checkpoint(new_model,epoch):
    model_out_path = "checkpoint/KC_model/" + "60per_KC_model_epoch_{}.pth".format(epoch)
    state = {"epoch" : epoch, "new_model": new_model}
    if not os.path.exists("checkpoint/KC_model/"):
        os.mkdirs("checkpoint/KC_model/")
    torch.save(state, model_out_path)
    print("checkpoint saved to {}".format(model_out_path))



opt = parser.parse_args()
cuda = opt.cuda

if cuda:
    print("=> use gpu id: '{}'".format(opt.gpus))
    os.environ["CUDA_VISIBLE_DEVICES"] = opt.gpus
    if not torch.cuda.is_available():
            raise Exception("No GPU found or Wrong gpu id, please run without --cuda")


model = torch.load(opt.model, map_location=lambda storage, loc: storage)["model"]

im_gt_ycbcr = imread(opt.image + ".bmp", mode="YCbCr")
im_b_ycbcr = imread(opt.image + "_scale_"+ str(opt.scale) + ".bmp", mode="YCbCr")
im_gt_y = im_gt_ycbcr[:,:,0].astype(float)
im_b_y = im_b_ycbcr[:,:,0].astype(float)




psnr_bicubic = PSNR(im_gt_y, im_b_y,shave_border=opt.scale)

im_input = im_b_y/255.

im_input = Variable(torch.from_numpy(im_input).float()).view(1, -1, im_input.shape[0], im_input.shape[1])
model = model.cpu()

start_time = time.time()
out = model(im_input)
elapsed_time = time.time() - start_time

out = out.cpu()


temp=list(model.parameters())
weights_sequence=[]
new_weights_sequence=[]
weights_sequence.append(temp[18].detach().numpy())
for lay in range(0,18):
    weights_sequence.append(temp[lay].detach().numpy())
weights_sequence.append(temp[19].detach().numpy())
new_weights_sequence,b=KC_Pruning(weights_sequence)
new_model=KC_Net(new_weights_sequence)
#print(new_weights_sequence[19])




if cuda:
    im_input = im_input.cuda()
    model=model.cuda()




print("===> Loading dataSets")
retrain_set = DatasetFromHdf5("data/train.h5")
retraining_data_loader = DataLoader(dataset=retrain_set, num_workers=opt.threads,batch_size=opt.batchSize, shuffle=True)

print("===> Building KC model")
criterion = nn.MSELoss(size_average=False)

print("===> Setting GPU")
if cuda:
    new_model = new_model.cuda()
    criterrion = criterion.cuda()

if opt.resume:
    if os.path.isfile(opt.resume):
        print("=> loading checkpoint '{}'".format(opt.resume))
        checkpoint = torch.load(opt.resume)
        opt.start_epoch = checkpoint["epoch"]+1
        new_model.load_state_dict(checkpoint["new_model"].state_dict())
    else:
        print("=> no checkpoint found at '{}'".format(opt.resume))
if opt.pretrained:
    if os.path.isfile(opt.pretrained):
        print("=> loading model'{}'".foramt(opt.pretrained))
        weights = torch.load(opt.pretrained)
        model.load_state_dict(weights["model"].state_dict())
    else:
        print("=> no model fount at '{}'".format(opt.pretrained))
print("===> Setting Optimizer")
optimizer = optim.SGD(new_model.parameters(),lr=opt.lr,  momentum=opt.momentum,weight_decay=opt.weight_decay)

print("===> ReTraining")
save_checkpoint(new_model,0)
#print(list(new_model.parameters())[19])
for epoch in range(opt.start_epoch, opt.nEpochs +1):
    train(retraining_data_loader, optimizer, new_model, criterion, epoch)
    save_checkpoint(new_model, epoch)

