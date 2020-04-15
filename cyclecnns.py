import torch
import torch.nn as nn
import torchvision.datasets as dsets
import torchvision.transforms as transforms
from torch.autograd import Variable
import scipy.io as sio
import os
from PIL import Image
import argparse
import torch.backends.cudnn as cudnn
import torch.utils.model_zoo as model_zoo
import numpy as np
import torch.nn.functional as F

#LEARNING_RATE = 0.01
#EPOCH = 5

parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=10000, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='enables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--pretrained', default=True,
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=1, metavar='N')
parser.add_argument('--log_interval_epoch', type=int, default=10, metavar='N')

args = parser.parse_args()
transform = None

#trainData = dsets.ImageFolder('../data/imagenet/train', transform)
#testData = dsets.ImageFolder('../data/imagenet/test', transform)

#trainLoader = torch.utils.data.DataLoader(dataset=trainData, batch_size=BATCH_SIZE, shuffle=True)
#testLoader = torch.utils.data.DataLoader(dataset=testData, batch_size=BATCH_SIZE, shuffle=False)


def default_image_loader(path):
    return Image.open(path)
#    return Image.open(path).convert('RGB')

class ImageLoader(torch.utils.data.Dataset):
    def __init__(self, base_path, trainVgg,trainMvcnn, transform=None,
                 loader=default_image_loader):
        """ filenames_filename: A text file with each line containing the path to an image e.g.,
                images/class1/sample.jpg
            triplets_file_name: A text file with each line containing three integers, 
                where integer i refers to the i-th image in the filenames file. 
                For a line of intergers 'a b c', a triplet is defined such that image a is more 
                similar to image c than it is to image b, e.g., 
                0 2017 42 """
        self.base_path = base_path  
        self.filenamelist = []
        for line in open(trainVgg):
            self.filenamelist.append(line.rstrip('\n'))
            
        self.filenamelist2 = []
        for line in open(trainMvcnn):
            self.filenamelist2.append(line.rstrip('\n'))

        self.transform = transform
        self.loader = loader

    def __getitem__(self, index):
        
        bais_path1 = os.path.join(self.base_path,self.filenamelist[int(index)])
        bais_path2 = os.path.join(self.base_path,self.filenamelist2[int(index)])
        
        vgg = sio.loadmat(bais_path1 + '.mat')
        mvcnn = sio.loadmat(bais_path2 + '.mat')
        
        
        vgg_o = vgg['shape'].astype(np.float32)
        vgg_o = torch.from_numpy(vgg_o)
        vgg_o = torch.squeeze(vgg_o)
        temp = np.reshape(vgg_o, (80,80))
        vgg = torch.Tensor(1,80,80)
        vgg[0,:,:] = temp
    
        
        mvcnn_o = mvcnn['shape'].astype(np.float32)
        mvcnn_o = torch.from_numpy(mvcnn_o)
        mvcnn_o = torch.squeeze(mvcnn_o)
        temp = np.reshape(mvcnn_o, (80,80))
        mvcnn = torch.Tensor(1,80,80)
        mvcnn[0,:,:] = mvcnn

        return vgg,mvcnn,vgg_o,mvcnn_o

    def __len__(self):
        return len(self.filenamelist)
		
class Classier(nn.Module):

    def __init__(self, num_classes=18):
        super(AlexNet, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(256 * 5 * 5, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.classifier(x)
        return x


class CycleCNN(nn.Module):
    def __init__(self):
        super(CycleCNN, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=3, padding=2),  # b, 16, 10, 10
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=2),  # b, 16, 5, 5
            nn.Conv2d(16, 8, 3, stride=3, padding=2),  # b, 8, 3, 3
            nn.ReLU(True),
            nn.MaxPool2d(2, stride=1)  # b, 8, 2, 2
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(8, 16, 5, stride=2),  # b, 16, 5, 5
            nn.ReLU(True),
            nn.ConvTranspose2d(16, 8, 5, stride=3, padding=1),  # b, 8, 15, 15
            nn.ReLU(True),
            nn.ConvTranspose2d(8, 1, 5, stride=2, padding=1,output_padding =1),  # b, 1, 28, 28
            nn.Tanh()
        )
 
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        
        length = len(x)
        temp = torch.Tensor(length,6400)
        for g in range(0,length):
            temp2 = x[g,0,:,:]
            temp3 = temp2.view(-1,6400)
            temp[g,:] = temp3
        return temp


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        

def train(vgg_mvcnn,mvcnn_vgg,vggc,mvcnnc,trainLoader,optimizer1,optimizer2,epoch,lRecord,criterion,cost):
    all_loss1 = AverageMeter()
    all_loss2 = AverageMeter()
    vgg_mvcnn.train()
    mvcnn_vgg.train()
	vggc.test()
	mvcnnc.test()
    for batch_idx, (vgg,mvcnn,vgg_o,mvcnn_o,labelvgg,labelmvcnn) in enumerate(trainLoader):
        if args.cuda:
            vgg,mvcnn = vgg.cuda(),mvcnn.cuda()
        vgg,mvcnn = Variable(vgg),Variable(mvcnn)
            
        # Forward + Backward + Optimize            
        feature = vgg_mvcnn(vgg)
    
        loss_1 = criterion(feature,mvcnn_o)
		loss_2 = criterion(mvcnn_vgg(feature),vgg_o)
		        
		labelvgg = np.transpose(labelvgg)
        labelvgg = torch.squeeze(labelvgg) 
        labelvgg = labelvgg.type(torch.LongTensor)
        labelvgg = labelvgg.cuda()
        labelvgg = Variable(labelvgg)
		
		loss_3 = cost(mvcnnc(feature),labelvgg)
		
		loss = loss_1+loss_2+loss_3
            
        loss.backward()
        optimizer1.step()
        optimizer1.zero_grad()
        
        all_loss1.update(loss.item(),1)
        
        feature = mvcnn_vgg(mvcnn)
        
        loss_1 = criterion(feature,vgg_o)
		loss_2 = criterion(vgg_mvcnn(feature),mvcnn_o)
		
		labelmvcnn = np.transpose(labelmvcnn)
        labelmvcnn = torch.squeeze(labelmvcnn) 
        labelmvcnn = labelmvcnn.type(torch.LongTensor)
        labelmvcnn = labelmvcnn.cuda()
        labelmvcnn = Variable(labelmvcnn)
		
		loss_3 = cost(vggc(feature),labelmvcnn)
		loss = loss_1 + loss_2 + loss_3
            
        loss.backward()
        optimizer2.step()
        optimizer2.zero_grad()
        
        all_loss2.update(loss.item(),1)
        
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{}]\t'
                  'Loss VGG_MVCNN: {:.4f} ({:.4f}) \t'
                  'Loss MVCNN_VGG: {:.4f} ({:.4f}) \t'.format(
                epoch, batch_idx * len(vgg), len(trainLoader.dataset),
                all_loss1.val, all_loss1.avg,
                all_loss2.val, all_loss2.avg))
            lRecord.append('Train Epoch: {} [{}/{}]\t'
                  'Loss VGG_MVCNN: {:.4f} ({:.4f}) \t'
                  'Loss MVCNN_VGG: {:.4f} ({:.4f}) \t'.format(
                epoch, batch_idx * len(vgg), len(trainLoader.dataset),
                all_loss1.val, all_loss1.avg,
                all_loss2.val, all_loss2.avg))
    
    return all_loss1.avg, all_loss2.avg


def main(datastr,step):
    
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    kwargs = {'num_workers': 0, 'pin_memory': True} if args.cuda else {}
    
    trainVgg = datastr + '/cycledata/train/image_vgg.txt'
    trainMvcnn = datastr + '/cycledata/train/image_mvcnn.txt'
    
    trainLoader = torch.utils.data.DataLoader(
            ImageLoader('../data',trainVgg,
                            trainMvcnn,
                            transform),
                            batch_size=args.batch_size, shuffle=True, **kwargs)
    
    vgg_mvcnn = CycleCNN()
    mvcnn_vgg = CycleCNN()
	
	vggc = Classier()
	mvcnnc = Classier()
    
    criterion = nn.MSELoss()
    if step > 0:        
        modelstr = datastr + '/model/model_' + str(step)+'vgg_mvcnn'
        modelstr = modelstr + '.pkl'
        vgg_mvcnn.load_state_dict(torch.load(modelstr))
        
        modelstr = datastr + '/model/model_' + str(step)+'mvcnn_vgg'
        modelstr = modelstr + '.pkl'
        mvcnn_vgg.load_state_dict(torch.load(modelstr))
		
		modelstr = datastr + '/model/model_' + str(step)+'vggc'
        modelstr = modelstr + '.pkl'
        vggc.load_state_dict(torch.load(modelstr))
		
		modelstr = datastr + '/model/model_' + str(step)+'mvcnnc'
        modelstr = modelstr + '.pkl'
        mvcnnc.load_state_dict(torch.load(modelstr))
		
    if args.cuda:
        vgg_mvcnn.cuda()
        mvcnn_vgg.cuda()
		vggc.cuda()
		mvcnnc.cuda()
		

    cudnn.benchmark = True
    # Loss and Optimizer

    optimizer1 = torch.optim.Adam(vgg_mvcnn.parameters(), lr=args.lr)
    optimizer2 = torch.optim.Adam(mvcnn_vgg.parameters(), lr=args.lr)
	cost = nn.CrossEntropyLoss()
    
    # Train the model
    lRecord = []
    loss_train1 = []
    loss_train2 = []
    for epoch in range(step + 1, args.epochs + step + 1):
        loss1,loss2 = train(vgg_mvcnn,mvcnn_vgg,vggc,mvcnnc,trainLoader,optimizer1,optimizer2,epoch,lRecord,criterion,cost)
        loss_train1.append(loss1)
        loss_train2.append(loss2)
        
        if epoch % args.log_interval_epoch == 0:
            directoy = datastr + '/model/'
            if os.path.exists(directoy) == False:
                os.mkdir(directoy)
        
            directoy = datastr + '/rate/'
            if os.path.exists(directoy) == False:
                os.mkdir(directoy)
        
            directoy = datastr + '/test/'
            if os.path.exists(directoy) == False:
                os.mkdir(directoy)
        
            directoy = datastr
    
            fileName = directoy + '/model/model_' + str(epoch) + 'vgg_mvcnn.pkl'
            torch.save(vgg_mvcnn.state_dict(), fileName)
            
            fileName = directoy + '/model/model_' + str(epoch) + 'mvcnn_vgg.pkl'
            torch.save(mvcnn_vgg.state_dict(), fileName)
             
            fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'loss_train_vgg.txt'
            f = open(fileName, 'w')
            for l in range(len(loss_train1)):
                f.write(str(loss_train1[l]))
                f.write('\n')
            f.close()
            
            fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'loss_train_mvcnn.txt'
            f = open(fileName, 'w')
            for l in range(len(loss_train2)):
                f.write(str(loss_train2[l]))
                f.write('\n')
            f.close()
            
            fileName = directoy + '/rate/model_' + str(epoch) + '_' + 'lRecord.txt'
            f = open(fileName, 'w')
            for i in range(len(lRecord)):
                f.write(lRecord[i])
                f.write('\n')
            f.close()
    


if __name__ == '__main__':

    dataSet = 'PSB'
    datastr = '' + dataSet

    step = 0
    main(datastr,step)

