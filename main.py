import argparse
import torch.backends.cudnn as cudnn
from scipy.io import loadmat
from model import GraphGST
import time
from utils import *
import scipy.io as io
import matplotlib.pyplot as plt

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

parser = argparse.ArgumentParser("HSI")
parser.add_argument('--dataset', choices=['Indian', 'WHU_Hi_LongKou', 'Houston2013', 'Houston2018'], default='Houston2013', help='dataset to use')
parser.add_argument('--gpu_id', default='0', help='gpu id')
parser.add_argument('--seed', type=int, default=0, help='number of seed')
parser.add_argument('--test_freq', type=int, default=50, help='number of evaluation')
parser.add_argument('--epoches', type=int, default=400, help='epoch number')
parser.add_argument('--variance', type=float, default=0.5, help='variance of RBF function')
parser.add_argument('--gamma', type=float, default=0.9, help='gamma')
parser.add_argument('--learning_rate', type=float, default= 5e-4, help='learning rate')
parser.add_argument('--weight_decay', type=float, default=0, help='weight_decay')
parser.add_argument('--batch_size', type=int, default=512, help='number of batch size')
parser.add_argument('--dropout', type=float, default=0.3, help='dropout_rate')
parser.add_argument('--window_a', type=int, default=7, help='window size of position matrix')
parser.add_argument('--window_b', type=int, default=11, help='number of patches')
parser.add_argument('--nlayers', type=int, default=4, help='number of network layers')
parser.add_argument('--r_positive', type=float, default=0.1, help='Threshold value')
parser.add_argument('--lambdas', type=float, default=0.1, help='weight of ss')


args = parser.parse_args()
np.random.seed(args.seed)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.deterministic = True
cudnn.benchmark = False

TR,TE,input,color_mat,num_classes,input_normalize = AcquireData(args)
row, column, dim = input.shape
Label_train,Label_test,num_patches, x_test_position = DataProcee(row, column, TR, TE, num_classes, dim, input_normalize, args)

print("Dataset:", args.dataset)
print("height={0},width={1},band={2}".format(row, column, dim))
print("patch size is : {}".format(args.window_a))
print("sliding window size is : {}".format(args.window_b))

label_train_loader=Data.DataLoader(Label_train,batch_size=args.batch_size,shuffle=True)
label_test_loader=Data.DataLoader(Label_test,batch_size=args.batch_size,shuffle=True)
# label_true_loader=Data.DataLoader(Label_true,batch_size=100,shuffle=False)

model = GraphGST(
    num_classes = num_classes,
    bonds=dim,
    nlayers = args.nlayers,
    heads = 4,
    num_position = num_patches,
    emb_dropout = args.dropout,
    r_positive = args.r_positive
)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.epoches//10, gamma=args.gamma)
#-------------------------------------------------------------------------------

print("start training")
tic = time.time()
for epoch in range(args.epoches):
    scheduler.step()
    model.train()
    # train_acc, train_obj, tar_t, pre_t = train_epoch(model, label_train_loader, optimizer)
    objs = AvgrageMeter()
    top1 = AvgrageMeter()
    tar = np.array([])
    pre = np.array([])
    for batch_idx, (batch_data, batch_data_position, batch_target) in enumerate(label_train_loader):
        batch_data = batch_data.to(device)
        batch_data_position = batch_data_position.to(device)
        batch_target = batch_target.to(device)
        optimizer.zero_grad()
        batch_pred, cross_view = model(batch_data, batch_data_position)
        loss = total_loss(batch_pred, batch_target, cross_view, args)
        loss.backward()
        optimizer.step()
        n = batch_data.shape[0]
        objs.update(loss.data, n)

    train_obj = objs.avg

    if (epoch+1) % 5 == 0:
        print("Epoch: {:03d} train_loss: {:.4f}"
              .format(epoch + 1, train_obj))
    if (epoch == args.epoches - 1):
    # if (epoch)%200==0|(epoch == args.epoches - 1):
        model.eval()
        tar_v, pre_v = valid_epoch(model, label_test_loader, x_test_position, optimizer, args)
        OA2, AA_mean2, Kappa2, AA2 = output_metric(tar_v, pre_v)
        print("epoch:",epoch)
        print("Final result:")
        print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
        print(AA2)

toc = time.time()
print("Running Time: {:.2f}".format(toc-tic))
print("=====================================================")

# print("Final result:")
# print("OA: {:.4f} | AA: {:.4f} | Kappa: {:.4f}".format(OA2, AA_mean2, Kappa2))
# print(AA2)
# print("=====================================================")
# print("Parameter:")

def print_args(args):
    for k, v in zip(args.keys(), args.values()):
        print("{0}: {1}".format(k,v))

print_args(vars(args))









