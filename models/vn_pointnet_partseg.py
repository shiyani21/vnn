import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from models.vn_layers import *
from models.vn_pointnet import STNkd
from models.utils.vn_dgcnn_util import get_graph_feature_cross


class get_model(nn.Module):
    def __init__(self, args, num_part=50, normal_channel=True):
        super(get_model, self).__init__()
        self.args = args
        self.n_knn = args.n_knn
        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.num_part = num_part
        
        self.conv_pos = VNLinearLeakyReLU(3, 64//3, dim=5, negative_slope=0.0)
        self.conv1 = VNLinearLeakyReLU(64//3, 64//3, dim=4, negative_slope=0.0)
        self.conv2 = VNLinearLeakyReLU(64//3, 128//3, dim=4, negative_slope=0.0)
        self.conv3 = VNLinearLeakyReLU(128//3, 128//3, dim=4, negative_slope=0.0)
        self.conv4 = VNLinearLeakyReLU(128//3*2, 512//3, dim=4, negative_slope=0.0)
        
        self.conv5 = VNLinear(512//3, 2048//3)
        self.bn5 = VNBatchNorm(2048//3, dim=4)
        
        self.std_feature = VNStdFeature(2048//3*2, dim=4, normalize_frame=False, negative_slope=0.0)
        
        if args.pooling == 'max':
            self.pool = VNMaxPool(64//3)
        elif args.pooling == 'mean':
            self.pool = mean_pool
        
        self.fstn = STNkd(args, d=128//3)
        
        self.convs1 = torch.nn.Conv1d(9025, 256, 1)
        self.convs2 = torch.nn.Conv1d(256, 256, 1)
        self.convs3 = torch.nn.Conv1d(256, 128, 1)
        self.convs4 = torch.nn.Conv1d(128, num_part, 1)
        self.bns1 = nn.BatchNorm1d(256)
        self.bns2 = nn.BatchNorm1d(256)
        self.bns3 = nn.BatchNorm1d(128)

        self.convs5 = torch.nn.Conv1d(50,num_part,1)
    def forward(self, point_cloud, label):
        B, D, N = point_cloud.size()
        print("this is B ",B)
        print("this is D ",D)
        print("this is N ",N)
        point_cloud = point_cloud.unsqueeze(1)
        feat = get_graph_feature_cross(point_cloud, k=self.n_knn)
        point_cloud = self.conv_pos(feat)
        point_cloud = self.pool(point_cloud)

        out1 = self.conv1(point_cloud)
        print("this is the first layer out1 ",out1.shape)
        out2 = self.conv2(out1)
        print("this is the 2nd layer out2 ",out2.shape)
        out3 = self.conv3(out2)
        print("this is the 3rd layer out3 ",out3.shape)

        net_global = self.fstn(out3).unsqueeze(-1).repeat(1,1,1,N)
        print("this is something called net_global ",net_global.shape)
        net_transformed = torch.cat((out3, net_global), 1)
        print("this is the net_transformed ",net_transformed.shape)

        out4 = self.conv4(net_transformed)
        print("this is the out4 ",out4.shape)
        out5 = self.bn5(self.conv5(out4))
        print("this is the bn5 aka out5 ",out5.shape)
        
        out5_mean = out5.mean(dim=-1, keepdim=True).expand(out5.size())
        print("some mean fn on out5 ",out5_mean.shape)
        out5 = torch.cat((out5, out5_mean), 1)
        print("some cat fn on out5 ",out5.shape)
        out5, trans = self.std_feature(out5)
        out5 = out5.view(B, -1, N)
        print("something else on out5 ",out5.shape)
        out_max = torch.max(out5, -1, keepdim=False)[0]

        out_max = torch.cat([out_max,label.squeeze(1)],1)
        expand = out_max.view(-1, 2048//3*6+16, 1).repeat(1, 1, N)
        
        out1234 = torch.cat((out1, out2, out3, out4), dim=1)
        out1234 = torch.einsum('bijm,bjkm->bikm', out1234, trans).view(B, -1, N)
        
        concat = torch.cat([expand, out1234, out5], 1)
        
        net = F.relu(self.bns1(self.convs1(concat)))
        print("net 1 ",net.shape)
        net = F.relu(self.bns2(self.convs2(net)))
        print("net 2",net.shape)
        net = F.relu(self.bns3(self.convs3(net)))
        print("net 3 ",net.shape)
        net = self.convs4(net)
        print("net 4",net.shape)
        net = self.convs5(net)
        print("this is mine, ",net.shape)
        #net = net.transpose(2, 1).contiguous()
#        net = self.conv2d(net,50,[1,1,1,1],"VALID")
#        net = torch.squeeze(net,[2])
        #print("net 5",net.shape)
        #net = F.log_softmax(net.view(-1, self.num_part), dim=-1)
        #print("net at softmax ",net.shape)
        #net = net.view(B, N, self.num_part) # [B, N, 50]
        #print("this is the type of net.view in the last layer ",type(net))
        #print("the size of tensor for last layer ",net.shape)
        trans_feat = None
        return net, trans_feat

    def get_activation(name):
        def hook(model,input,ouput):
            activation[name]=output.detach()
        return hook

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        return loss

#model = get_model()
#model.bns3.register_forward_hook(get_activation('bns3'))
#x=torch.randn(1,25)
#output=model(x)
#print(activation['bns3'])

