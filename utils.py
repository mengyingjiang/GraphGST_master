import torch
import torch.nn as nn
import math
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
import numpy as np
import os
import torch.utils.data as Data
from scipy.io import loadmat

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def AcquireData(args):
    if args.dataset == 'Indian':
        data = loadmat('./data/IndianPine.mat')
        input = data['input']
        TR = data['TR']
        TE = data['TE']


    elif args.dataset == 'Houston2018':
        Houston2018 = loadmat('./data/Houston2018/DFC2018.mat')
        TR = loadmat('./data/Houston2018/TR.mat')
        TE = loadmat('./data/Houston2018/TE.mat')
        input = Houston2018['DFC2018']
        TR = TR['TR']
        TE = TE['TE']

    elif args.dataset == 'Houston2013':
        data = loadmat('./data/Houston.mat')
        TR = data['TR']
        TE = data['TE']
        input = data['input']

    elif args.dataset == 'WHU_Hi_LongKou':
        WHU_Hi_LongKou = loadmat('./data/WHU/WHU_Hi_LongKou.mat')
        TR = loadmat('./data/WHU/TR.mat')
        TE = loadmat('./data/WHU/TE.mat')
        input = WHU_Hi_LongKou['WHU_Hi_LongKou']
        TR = TR['TR']
        TE = TE['TE']

    else:
        raise ValueError("Unkknow dataset")
    color_mat = loadmat('./data/AVIRIS_colormap.mat')
    num_classes = np.max(TR)

    input_normalize = np.zeros(input.shape)
    for i in range(input.shape[2]):
        input_max = np.max(input[:, :, i])
        input_min = np.min(input[:, :, i])
        input_normalize[:, :, i] = (input[:, :, i] - input_min) / (input_max - input_min)

    return TR,TE,input,color_mat,num_classes,input_normalize


def DataProcee(height, width,TR, TE, num_classes,band, input_normalize, args):
    label = TR + TE
    patch_labels, num_patches, nr, nc = position_matri(height, width, args)  # divide a image into many patches.

    # obtain train and test position
    pos_train, pos_test, pos_true, num_train, num_test, num_true = \
        chooose_train_and_test_point(TR, TE, label, num_classes)

    # padding
    mirror_image = mirror_hsi(height, width, band, input_normalize, patches_real=args.window_b)  #

    # obtain train and test features
    x_train_band, x_test_band, x_true_band, x_train_position, x_test_position = \
        train_and_test_data(mirror_image, band, pos_train, pos_test, pos_true, patch_labels, nr, nc, num_patches, args)

    y_train, y_test, y_true = train_and_test_label(num_train, num_test, num_true, num_classes)
    # -------------------------------------------------------------------------------
    # load data
    x_train = torch.from_numpy(x_train_band).type(torch.FloatTensor)
    x_train_position = torch.from_numpy(x_train_position).type(torch.FloatTensor)
    y_train = torch.from_numpy(y_train).type(torch.LongTensor)
    Label_train = Data.TensorDataset(x_train, x_train_position, y_train)

    x_test = torch.from_numpy(x_test_band).type(torch.FloatTensor)
    # x_test_position = torch.from_numpy(x_test_position).type(torch.FloatTensor)
    y_test = torch.from_numpy(y_test).type(torch.LongTensor)
    x_test_indexs = torch.arange(x_test_position.shape[0])
    Label_test = Data.TensorDataset(x_test,x_test_indexs, y_test)
    # Label_test = Data.TensorDataset(x_test, x_test_position, y_test)

    # x_true=torch.from_numpy(x_true_band).type(torch.FloatTensor)
    # x_true_position=torch.from_numpy(x_true_position).type(torch.FloatTensor)
    # y_true=torch.from_numpy(y_true).type(torch.LongTensor)
    # Label_true=Data.TensorDataset(x_true,x_true_position,y_true)

    return Label_train,Label_test,num_patches, x_test_position

#-------------------------------------------------------------------------------
# 定位训练和测试样本
def chooose_train_and_test_point(train_data, test_data, true_data, num_classes):
    number_train = []
    pos_train = {}
    number_test = []
    pos_test = {}
    number_true = []
    pos_true = {}
    #-------------------------for train data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(train_data==(i+1))
        number_train.append(each_class.shape[0])
        pos_train[i] = each_class

    total_pos_train = pos_train[0]
    for i in range(1, num_classes):
        total_pos_train = np.r_[total_pos_train, pos_train[i]] #(695,2)
    total_pos_train = total_pos_train.astype(int)
    #--------------------------for test data------------------------------------
    for i in range(num_classes):
        each_class = []
        each_class = np.argwhere(test_data==(i+1))
        number_test.append(each_class.shape[0])
        pos_test[i] = each_class

    total_pos_test = pos_test[0]
    for i in range(1, num_classes):
        total_pos_test = np.r_[total_pos_test, pos_test[i]] #(9671,2)
    total_pos_test = total_pos_test.astype(int)
    #--------------------------for true data------------------------------------
    for i in range(num_classes+1):
        each_class = []
        each_class = np.argwhere(true_data==i)
        number_true.append(each_class.shape[0])
        pos_true[i] = each_class

    total_pos_true = pos_true[0]
    for i in range(1, num_classes+1):
        total_pos_true = np.r_[total_pos_true, pos_true[i]]
    total_pos_true = total_pos_true.astype(int)

    return total_pos_train, total_pos_test, total_pos_true, number_train, number_test, number_true
#-------------------------------------------------------------------------------
# 边界拓展：镜像
def mirror_hsi(height,width,band,input_normalize,patches_real):
    padding=patches_real//2
    mirror_hsi=np.zeros((height+2*padding,width+2*padding,band),dtype=float)
    #中心区域
    mirror_hsi[padding:(padding+height),padding:(padding+width),:]=input_normalize
    # 左边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),i,:]=input_normalize[:,padding-i-1,:]
    #右边镜像
    for i in range(padding):
        mirror_hsi[padding:(height+padding),width+padding+i,:]=input_normalize[:,width-1-i,:]
    #上边镜像
    for i in range(padding):
        mirror_hsi[i,:,:]=mirror_hsi[padding*2-i-1,:,:]
    #下边镜像
    for i in range(padding):
        mirror_hsi[height+padding+i,:,:]=mirror_hsi[height+padding-1-i,:,:]

    print("mirror_image shape : [{0},{1},{2}]".format(mirror_hsi.shape[0],mirror_hsi.shape[1],mirror_hsi.shape[2]))
    print("=====================================================")
    return mirror_hsi
#-------------------------------------------------------------------------------
# 获取patch的图像数据
def gain_neighborhood_pixel(mirror_image, point, i, patches_real,args):
    variance = args.variance
    patch = patches_real
    wc = (patch - 1) // 2
    x = point[i,0]
    y = point[i,1]
    temp_image = mirror_image[x:(x+patch),y:(y+patch),:]
    temp_image_in = temp_image.reshape(patch ** 2, -1)
    center = mirror_image[x+wc,y+wc,:]
    temp_image = temp_image_in-center
    A =  np.sum( temp_image * temp_image, 1)
    we = (np.exp(-variance*A))
    we = (we/(we.sum()))
    temp_image = (np.matmul(we, temp_image_in))

    return temp_image
    # return np.concatenate((temp_image,center),0)

def position_matri(r,c, args):
    window = args.window_a
    length = args.window_b
    n_r =  math.ceil(r / window)
    n_c =  math.ceil(c / window)
    position_matrix = np.zeros((r,c))
    i=1
    for x in range(n_r):
        for y in range(n_c):
            if x<(n_r-1):
                if y<(n_c-1):
                    position_matrix[x*window:(x*window + window), y*window:(y*window + window)] = i
                    i = i+1
                else:
                    position_matrix[x*window:(x*window + window), y*window:] = i
                    i = i+1
            else:
                if y<(n_c-1):
                    position_matrix[x*window: , y*window:(y*window + window)] = i
                    i = i+1
                else:
                    position_matrix[x*window: , y*window:] = i
                    i = i+1
    num_position = n_r * n_c
    half=(length-1)//2
    position_fea = np.zeros((r+length, c+length))

    position_fea[half:half+r,half:half+c]=position_matrix

    return position_matrix, num_position, n_r, n_c

def position_fea(position_matrix,n_r,n_c,point,i,args):
    length = args.window_b
    position_vector = np.zeros((n_r * n_c))
    x = point[i,0]
    y = point[i,1]
    window_num = position_matrix[x:x+length, y:y +length]
    window_num = window_num.reshape(1,-1)
    window_num = window_num[window_num!=0]-1
    unique_num, unique_count = np.unique(window_num,return_counts=True)
    unique_num = np.array(unique_num, dtype=int)
    position_vector[unique_num] = unique_count
    return position_vector

#-------------------------------------------------------------------------------
# 汇总训练数据和测试数据
def train_and_test_data(mirror_image, band, train_point, test_point, true_point,  position_matrix, n_r,n_c, num_position, args):
    patches_real = args.window_b
    x_train = np.zeros((train_point.shape[0], band), dtype=float)
    x_test = np.zeros((test_point.shape[0],band), dtype=float)
    x_true = np.zeros((true_point.shape[0], band), dtype=float)

    x_train_position = np.zeros((train_point.shape[0], num_position), dtype=float)
    x_test_position = np.zeros((test_point.shape[0], num_position), dtype=float)
    # x_true_position = np.zeros((true_point.shape[0], num_position), dtype=float)

    for i in range(train_point.shape[0]):
        x_train[i,:] = gain_neighborhood_pixel(mirror_image, train_point, i, patches_real,args)
    for j in range(test_point.shape[0]):
        x_test[j,:] = gain_neighborhood_pixel(mirror_image, test_point, j,patches_real,args)
    for k in range(true_point.shape[0]):
        x_true[k,:] = gain_neighborhood_pixel(mirror_image, true_point, k, patches_real,args)

    for i in range(train_point.shape[0]):
        x_train_position[i,:] = position_fea(position_matrix, n_r,n_c,train_point,i,args)
    for j in range(test_point.shape[0]):
        x_test_position[j,:] = position_fea(position_matrix, n_r,n_c,test_point,j,args)
    # for k in range(true_point.shape[0]):
    #     x_true_position[k,:] = position_fea(position_matrix, n_r,n_c,true_point,k,args)

    # x_train_position = x_train
    # x_test_position = x_test
    # x_true_position = x_true

    print("x_train shape = {}".format(x_train.shape))
    print("x_test  shape = {}".format(x_test.shape))
    print("x_true  shape = {}".format(x_true.shape))
    print("=====================================================")
    return x_train, x_test, x_true, x_train_position, x_test_position
    # return x_train, x_test, x_true, x_train_position, x_test_position, x_true_position
#-------------------------------------------------------------------------------
# 标签y_train, y_test
def train_and_test_label(number_train, number_test, number_true, num_classes):
    y_train = []
    y_test = []
    y_true = []
    for i in range(num_classes):
        for j in range(number_train[i]):
            y_train.append(i)
        for k in range(number_test[i]):
            y_test.append(i)
    for i in range(num_classes+1):
        for j in range(number_true[i]):
            y_true.append(i)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    y_true = np.array(y_true)
    # print("y_train: shape = {} ,type = {}".format(y_train.shape,y_train.dtype))
    # print("y_test: shape = {} ,type = {}".format(y_test.shape,y_test.dtype))
    # print("y_true: shape = {} ,type = {}".format(y_true.shape,y_true.dtype))
    # print("**************************************************")
    return y_train, y_test, y_true
#-------------------------------------------------------------------------------

class AvgrageMeter(object):

    def __init__(self):
        self.reset()

    def reset(self):
        self.avg = 0
        self.sum = 0
        self.cnt = 0

    def update(self, val, n=1):
        self.sum += val * n
        self.cnt += n
        self.avg = self.sum / self.cnt


# -------------------------------------------------------------------------------
def accuracy(output, target, topk=(1,)):
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res, target, pred.squeeze()


# -------------------------------------------------------------------------------
# train model
def train_epoch(model, train_loader, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_data_position, batch_target) in enumerate(train_loader):
        batch_data = batch_data.to(device)
        batch_data_position = batch_data_position.to(device)
        batch_target = batch_target.to(device)
        optimizer.zero_grad()
        batch_pred, cross_view = model(batch_data, batch_data_position)
        loss = total_loss(batch_pred, batch_target, cross_view)
        loss.backward()
        optimizer.step()
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())
    return top1.avg, objs.avg, tar, pre


# -------------------------------------------------------------------------------
# validate model
def valid_epoch(model, valid_loader, x_test_position,  optimizer, args):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_data_position, batch_target) in enumerate(valid_loader):
        batch_data = batch_data.to(device)
        batch_data_position = np.array(batch_data_position)
        batch_data_position = torch.from_numpy(x_test_position[batch_data_position,:]).type(torch.FloatTensor)
        batch_data_position = batch_data_position.to(device)
        batch_target = batch_target.to(device)
        batch_pred, cross_view = model(batch_data, batch_data_position)
        loss = total_loss(batch_pred, batch_target, cross_view, args)
        prec1, t, p = accuracy(batch_pred, batch_target, topk=(1,))
        n = batch_data.shape[0]
        objs.update(loss.data, n)
        top1.update(prec1[0].data, n)
        tar = np.append(tar, t.data.cpu().numpy())
        pre = np.append(pre, p.data.cpu().numpy())

    return tar, pre


def test_epoch(model, test_loader, optimizer):
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_data_position, batch_target) in enumerate(test_loader):
        batch_data = batch_data.to(device)
        batch_data_position = batch_data_position.to(device)
        batch_target = batch_target.to(device)

        batch_pred, cross_view = model(batch_data, batch_data_position)

        _, pred = batch_pred.topk(1, 1, True, True)
        pp = pred.squeeze()
        pre = np.append(pre, pp.data.cpu().numpy())
    return pre


# -------------------------------------------------------------------------------
def output_metric(tar, pre):
    matrix = confusion_matrix(tar, pre)
    OA, AA_mean, Kappa, AA = cal_results(matrix)
    return OA, AA_mean, Kappa, AA


# -------------------------------------------------------------------------------
def total_loss(batch_pred, batch_target, loss_ss, args):
    criterion = nn.CrossEntropyLoss().to(device)
    loss = criterion(batch_pred, batch_target)
    return loss_ss + (args.lambdas*loss)

# -------------------------------------------------------------------------------
def cal_results(matrix):
    shape = np.shape(matrix)
    number = 0
    sum = 0
    AA = np.zeros([shape[0]], dtype=float)
    for i in range(shape[0]):
        number += matrix[i, i]
        AA[i] = matrix[i, i] / np.sum(matrix[i, :])
        sum += np.sum(matrix[i, :]) * np.sum(matrix[:, i])
    OA = number / np.sum(matrix)
    AA_mean = np.mean(AA)
    pe = sum / (np.sum(matrix) ** 2)
    Kappa = (OA - pe) / (1 - pe)
    return OA, AA_mean, Kappa, AA
# -------------------------------------------------------------------------------