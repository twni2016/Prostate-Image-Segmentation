# Baseline Code
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt
import math

if __name__ == '__main__':

	# argparse settings
	import argparse
	parser = argparse.ArgumentParser(description='PROS12')
	parser.add_argument('-b', '--batch', type=int, default=16, help='input batch size for training (default: 64)')
	parser.add_argument('-e', '--epoch', type=int, default=55, help='number of epochs to train (default: 50)')
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
	parser.add_argument('--gpu', type=int, default=2, help='GPU (default: 4)')
	args = parser.parse_args()


	# HyperParameter
	epoch = args.epoch
	batch_size = args.batch
	lr = args.lr
	gpu_list = [item for item in range(args.gpu)]


	from datetime import datetime
	start = datetime.now()

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,),(0.5,)) # tuple for one channel
	])

	from dataset2d import PROS12
	training_set = PROS12(train=True, transform=transform)
	testing_set = PROS12(train=False,transform=transform)

	trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
	testloader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=True)


def to_var(x, volatile):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x,volatile=volatile)



# class DownTransition(nn.Module):

#     def __init__(self,inchan,layer,dilation_=1):
#         super(DownTransition, self).__init__()
#         self.dilation_ = dilation_
#         if inchan == 1: 
#             self.outchan = 8
#         else:
#             self.outchan = 2*inchan
#         self.layer = layer
#         self.down = nn.Conv2d(in_channels=inchan,out_channels=self.outchan,kernel_size=3,padding=1,stride=2) # /2
#         self.bn = nn.BatchNorm2d(num_features=self.outchan,affine=True)
#         self.conv = self.make_layers()
#         self.relu = nn.ELU(inplace=True)

#     def make_layers(self):

#         layers = []
#         for i in range(self.layer):
#             layers.append(nn.ELU(inplace=True))
#             # padding = dilation
#             layers.append(nn.Conv2d(self.outchan,self.outchan,kernel_size=3,padding=self.dilation_,stride=1,dilation=self.dilation_))
#             layers.append(nn.BatchNorm2d(num_features=self.outchan,affine=True))
            
#         return nn.Sequential(*layers)

#     def forward(self,x):

#         out1 = self.down(x)
#         out2 = self.conv(self.bn(out1))
#         out2 = self.relu(torch.add(out1,out2))
#         return out2

# class UpTransition(nn.Module):

#     def __init__(self,inchan,layer,last=False):
#         super(UpTransition, self).__init__()
#         self.last = last
#         self.outchan = inchan//2
#         self.layer = layer
#         self.up =  nn.ConvTranspose2d(in_channels=inchan,out_channels=self.outchan,kernel_size=4,padding=1,stride=2) # *2
#         self.bn = nn.BatchNorm2d(num_features=self.outchan,affine=True)
#         self.conv = self.make_layers()
#         self.relu = nn.ELU(inplace=True)
#         if self.last is True:
#             self.conv1 = nn.Conv2d(self.outchan,2,kernel_size=1) # 1*1 conv
#             self.softmax = F.softmax

#     def make_layers(self):

#         layers = []
#         for i in range(self.layer):
#             layers.append(nn.ELU(inplace=True))
#             layers.append(nn.Conv2d(self.outchan,self.outchan,kernel_size=3,padding=1,stride=1))
#             layers.append(nn.BatchNorm2d(num_features=self.outchan,affine=True))
            
#         return nn.Sequential(*layers)

#     def forward(self,x):

#         out1 = self.up(x)
#         out = self.conv(self.bn(out1))
#         out = self.relu(torch.add(out1,out))
#         if self.last is True:
#             out = self.conv1(out)
#             out = out.permute(0, 2, 3, 1).contiguous()
#             # print('forward',out.shape)
#             # flatten to (N,HW,C=2)
#             # out = out.view(out.size(0),-1,2)
#             # out = self.softmax(out,dim=2)
#             # out = torch.max(out,dim=2)[1].float()
#             # print('softmax',out.shape)
#             # result (N,HW)
#             out = out.view(out.numel() // 2, 2)
#             out = self.softmax(out,dim=1) # default
#         return out 


class down_transition(nn.Module):
    def __init__(self, input_channels, keep_prob):
        super(down_transition, self).__init__()
        self.input_channels = input_channels
        self.keep_prob = keep_prob
        self.bn = nn.BatchNorm2d(self.input_channels)
        self.conv = nn.Conv2d(self.input_channels, self.input_channels, kernel_size=3, padding=1, stride=2, bias = False) # /2
        # self.dropout = nn.Dropout2d(1 - self.keep_prob)
        # self.pool = nn.AvgPool2d(kernel_size = 2)

    def forward(self, x):
        output = self.bn(x)
        output = F.relu(output)
        output = self.conv(output)
        # output = self.dropout(output)
        # output = self.pool(output)
        return output


class up_transition(nn.Module):
    def __init__(self, input_channels, keep_prob, last=False):
        super(up_transition, self).__init__()
        self.last = last
        self.input_channels = input_channels
        self.keep_prob = keep_prob
        self.bn = nn.BatchNorm2d(self.input_channels)
        self.deconv = nn.ConvTranspose2d(self.input_channels, self.input_channels, kernel_size=4, padding=1, stride=2, bias = False) # *2
        # self.dropout = nn.Dropout2d(1 - self.keep_prob)
        # self.pool = nn.AvgPool2d(kernel_size = 2)
        if self.last is True:
            self.conv1 = nn.Conv2d(self.input_channels, 2, kernel_size=1) # 1*1 conv
            self.softmax = F.softmax

    def forward(self, x):
        output = self.bn(x)
        output = F.relu(output)
        output = self.deconv(output)
        # output = self.dropout(output)
        # output = self.pool(output)
        if self.last is True:
            output = self.conv1(output)
            output = output.permute(0, 2, 3, 1).contiguous()
            # print('forward',out.shape)
            # flatten to (N,HW,C=2)
            # out = out.view(out.size(0),-1,2)
            # out = self.softmax(out,dim=2)
            # out = torch.max(out,dim=2)[1].float()
            # print('softmax',out.shape)
            # result (N,HW)
            output = output.view(output.numel() // 2, 2)
            output = self.softmax(output, dim=1) # default
        return output


# class global_pool(nn.Module):
#     def __init__(self, input_size, input_channels):
#         super(global_pool, self).__init__()
#         self.input_size = input_size
#         self.input_channels = input_channels
#         self.bn = nn.BatchNorm2d(self.input_channels)
#         self.pool = nn.AvgPool2d(kernel_size = self.input_size)

#     def forward(self, x):
#         output = self.bn(x)
#         output = F.relu(output)
#         output = self.pool(output)
#         return output

# class compress(nn.Module):
#     def __init__(self, input_channels, keep_prob):
#         super(compress, self).__init__()
#         self.keep_prob = keep_prob
#         self.bn = nn.BatchNorm2d(input_channels)
#         self.conv = nn.Conv2d(input_channels, input_channels//2, kernel_size = 1, padding = 0, bias = False)
        

#     def forward(self, x):
#         output = self.bn(x)
#         output = F.relu(output)
#         output = self.conv(output)
#         # output = F.dropout2d(output, 1 - self.keep_prob)
#         return output

class Clique_block(nn.Module):
    def __init__(self, input_channels, channels_per_layer, layer_num, loop_num, keep_prob, dilation):
        super(Clique_block, self).__init__()
        self.input_channels = input_channels
        self.channels_per_layer = channels_per_layer
        self.layer_num = layer_num
        self.loop_num = loop_num
        self.keep_prob = keep_prob
        self.dilation = dilation

        # conv 1 x 1
        self.conv_param = nn.ModuleList([nn.Conv2d(self.channels_per_layer, self.channels_per_layer, kernel_size = 1, padding = 0, bias = False) 
                                   for i in range((self.layer_num + 1) ** 2)])

        for i in range(1, self.layer_num + 1):
            self.conv_param[i] = nn.Conv2d(self.input_channels, self.channels_per_layer, kernel_size = 1, padding = 0, bias = False)
        for i in range(1, self.layer_num + 1):
            self.conv_param[i * (self.layer_num + 2)] = None
        for i in range(0, self.layer_num + 1):
            self.conv_param[i * (self.layer_num + 1)] = None

        self.forward_bn = nn.ModuleList([nn.BatchNorm2d(self.input_channels + i * self.channels_per_layer) for i in range(self.layer_num)])
        self.forward_bn_b = nn.ModuleList([nn.BatchNorm2d(self.channels_per_layer) for i in range(self.layer_num)])
        self.loop_bn = nn.ModuleList([nn.BatchNorm2d(self.channels_per_layer * (self.layer_num - 1)) for i in range(self.layer_num)])
        self.loop_bn_b = nn.ModuleList([nn.BatchNorm2d(self.channels_per_layer) for i in range(self.layer_num)])

        # conv 3 x 3
        self.conv_param_bottle = nn.ModuleList([nn.Conv2d(self.channels_per_layer, self.channels_per_layer, kernel_size = 3, padding = self.dilation, bias = False, dilation = self.dilation) 
                                   for i in range(self.layer_num)])


    def forward(self, x):
        # key: 1, 2, 3, 4, 5, update every loop
        self.blob_dict={}
        # save every loops results
        self.blob_dict_list=[]

        # first forward
        for layer_id in range(1, self.layer_num + 1):
            bottom_blob = x
            # bottom_param = self.param_dict['0_' + str(layer_id)]

            bottom_param = self.conv_param[layer_id].weight
            for layer_id_id in range(1, layer_id):
                # pdb.set_trace()
                bottom_blob = torch.cat((bottom_blob, self.blob_dict[str(layer_id_id)]), 1)
                # bottom_param = torch.cat((bottom_param, self.param_dict[str(layer_id_id) + '_' + str(layer_id)]), 1)
                bottom_param = torch.cat((bottom_param, self.conv_param[layer_id_id * (self.layer_num + 1) + layer_id].weight), 1)
            next_layer = self.forward_bn[layer_id - 1](bottom_blob)
            next_layer = F.relu(next_layer)
            # conv 1 x 1
            next_layer = F.conv2d(next_layer, bottom_param, stride = 1, padding = 0)
            # conv 3 x 3
            next_layer = self.forward_bn_b[layer_id - 1](next_layer)
            next_layer = F.relu(next_layer)
            next_layer = F.conv2d(next_layer, self.conv_param_bottle[layer_id - 1].weight, stride = 1, padding = self.dilation, dilation = self.dilation)
            # next_layer = F.dropout2d(next_layer, 1 - self.keep_prob)
            self.blob_dict[str(layer_id)] = next_layer
        self.blob_dict_list.append(self.blob_dict)

        # loop
        for loop_id in range(self.loop_num):
            for layer_id in range(1, self.layer_num + 1): 
                
                layer_list = [l_id for l_id in range(1, self.layer_num + 1)]
                layer_list.remove(layer_id)
                
                bottom_blobs = self.blob_dict[str(layer_list[0])]
                # bottom_param = self.param_dict[layer_list[0] + '_' + str(layer_id)]
                bottom_param = self.conv_param[layer_list[0] * (self.layer_num + 1) + layer_id].weight
                for bottom_id in range(len(layer_list) - 1):
                    bottom_blobs = torch.cat((bottom_blobs, self.blob_dict[str(layer_list[bottom_id + 1])]), 1)
                    # bottom_param = torch.cat((bottom_param, self.param_dict[layer_list[bottom_id+1]+'_'+str(layer_id)]), 1)
                    bottom_param = torch.cat((bottom_param, self.conv_param[layer_list[bottom_id + 1] * (self.layer_num + 1) + layer_id].weight), 1) 
                bottom_blobs = self.loop_bn[layer_id - 1](bottom_blobs)
                bottom_blobs = F.relu(bottom_blobs)
                # conv 1 x 1
                mid_blobs = F.conv2d(bottom_blobs, bottom_param, stride = 1, padding = 0)
                # conv 3 x 3
                top_blob = self.loop_bn_b[layer_id - 1](mid_blobs)
                top_blob = F.relu(top_blob)
                top_blob = F.conv2d(top_blob, self.conv_param_bottle[layer_id - 1].weight, stride = 1, padding = self.dilation, dilation = self.dilation)
                self.blob_dict[str(layer_id)] = top_blob
            self.blob_dict_list.append(self.blob_dict)
        
        assert len(self.blob_dict_list) == 1 + self.loop_num

        # output
        # block_feature_I = self.blob_dict_list[0]['1']
        # for layer_id in range(2, self.layer_num + 1):
        #     block_feature_I = torch.cat((block_feature_I, self.blob_dict_list[0][str(layer_id)]), 1)
        # block_feature_I = torch.cat((x, block_feature_I), 1)
        
        block_feature_II = self.blob_dict_list[self.loop_num]['1']
        for layer_id in range(2, self.layer_num + 1):
            block_feature_II = torch.cat((block_feature_II, self.blob_dict_list[1][str(layer_id)]), 1)    
        # return block_feature_I, block_feature_II
        return block_feature_II

class CliqueNet(nn.Module):
    def __init__(self, input_channels, list_channels, list_layer_num):
        super(CliqueNet, self).__init__()
        self.fir_trans = nn.Conv2d(1, input_channels, kernel_size=7, stride=2, padding=3, bias=False) # conv 7*7 /2 = 256*256
        self.fir_bn = nn.BatchNorm2d(input_channels)
        # self.fir_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1) # /2 = 128*128
        self.block_num = len(list_channels) 

        self.down_block = nn.ModuleList()
        self.down_trans = nn.ModuleList()
        self.up_block = nn.ModuleList()
        self.up_trans = nn.ModuleList()
        # self.list_gb = nn.ModuleList()
        # self.list_gb_channel = []
        # self.list_compress = nn.ModuleList()
        input_size_init = 256

        for i in range(self.block_num): 
            if i == 0:
                self.down_block.append(Clique_block(input_channels=input_channels, channels_per_layer=list_channels[0], layer_num=list_layer_num[0], loop_num=1, keep_prob=0.8, dilation=2)) # 
                # self.list_gb_channel.append(input_channels + list_channels[0] * list_layer_num[0])
            else :
                self.down_block.append(Clique_block(input_channels=list_channels[i-1] * list_layer_num[i-1], channels_per_layer=list_channels[i], layer_num=list_layer_num[i], loop_num=1, keep_prob=0.8, dilation=2))
                # self.list_gb_channel.append(list_channels[i-1] * list_layer_num[i-1] + list_channels[i] * list_layer_num[i])

            if i < self.block_num - 1:
                self.down_trans.append(down_transition(input_channels=list_channels[i] * list_layer_num[i], keep_prob=0.8))

        for i in range(self.block_num):
            if i == 0:
                self.up_trans.append(up_transition(input_channels=list_channels[i] * list_layer_num[i], keep_prob=0.8, last=True))
            else : 
                self.up_trans.append(up_transition(input_channels=list_channels[i] * list_layer_num[i], keep_prob=0.8))

            if i>0 :
                self.up_block.append(Clique_block(input_channels=list_channels[i] * list_layer_num[i], channels_per_layer=list_channels[i-1], layer_num=list_layer_num[i-1], loop_num=1, keep_prob=0.8, dilation=2))
            else:
                self.up_block.append(None)
                # self.list_gb_channel.append(list_channels[i-1] * list_layer_num[i-1] + list_channels[i] * list_layer_num[i])


        #     self.list_gb.append(global_pool(input_size=input_size_init, input_channels=list_gb_channel[i] // 2))
        #     self.list_compress.append(compress(input_channels=list_gb_channel[i], keep_prob=0.8))
        #     input_size_init = input_size_init // 2

        #  self.fc = nn.Linear(in_features=sum(list_gb_channel) // 2, out_features=1000)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):

        output = self.fir_trans(x)
        output = self.fir_bn(output)
        output = F.relu(output)
        # output = self.fir_pool(output)

        feature_II_down_list = []

        # use stage II + stage II mode 
        for i in range(self.block_num):
            block_feature_II = self.down_block[i](output) 
            feature_II_down_list.append(block_feature_II)
            # print(i,'down_feature_I',block_feature_I.shape,'feature_II',block_feature_II.shape)
            # block_feature_I = self.list_compress[i](block_feature_I)
            # feature_I_list.append(self.list_gb[i](block_feature_I))
            if i < self.block_num - 1:
                output = self.down_trans[i](block_feature_II) # 

        for i in reversed(range(self.block_num)):
            if i == self.block_num - 1:
                output = self.up_trans[i](feature_II_down_list[i])
            else:
                output = self.up_trans[i](block_feature_II+feature_II_down_list[i])

            # print(i,'up_trans',output.shape)
            if i>0 :
                block_feature_II = self.up_block[i](output) 
                # print(i,'up_feature_I',block_feature_I.shape,'feature_II',block_feature_II.shape)
            # block_feature_I = self.list_compress[i](block_feature_I)
            # feature_I_list.append(self.list_gb[i](block_feature_I))


        # final_feature = feature_I_list[0]
        # for block_id in range(1, len(feature_I_list)):
        #     final_feature=torch.cat((final_feature, feature_I_list[block_id]), 1)
        
        # final_feature = final_feature.view(final_feature.size()[0], final_feature.size()[1])
        # output = self.fc(final_feature)
        return output





# class Vnet(nn.Module):
# 	# 1*512*512
# 	def __init__(self):
# 		super(Vnet,self).__init__()

# 		self.layer0 = nn.Sequential(
# 				nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=3,bias=False),
# 				nn.BatchNorm2d(8,affine=True),
# 				nn.ELU(inplace=True)
# 			)
# 		self.down0 = DownTransition(inchan=8,layer=2,dilation_=2)  # 16*256^2
# 		self.down1 = DownTransition(inchan=16,layer=2,dilation_=2) # 32*128^2
# 		self.down2 = DownTransition(inchan=32,layer=2,dilation_=2) # 64*64^2
# 		self.down3 = DownTransition(inchan=64,layer=2,dilation_=4) # 128*32^2
		

# 		self.up3 = UpTransition(inchan=128,layer=2) # 64*64^2
# 		self.up2 = UpTransition(inchan=64,layer=2) # 32*128^2
# 		self.up1 = UpTransition(inchan=32,layer=2) # 16*256^2
# 		self.up0 = UpTransition(inchan=16,layer=2,last=True) # 2*512^2

# 		for m in self.modules():
# 			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
# 				nn.init.kaiming_normal(m.weight.data)
# 			elif isinstance(m, nn.BatchNorm2d):
# 				m.weight.data.fill_(1)
# 				m.bias.data.zero_()

# 	def forward(self,x):
		
# 		x = self.layer0(x)
# 		out_down0 = self.down0(x)        
# 		out_down1 = self.down1(out_down0)
# 		out_down2 = self.down2(out_down1) 
# 		out_down3 = self.down3(out_down2) 

# 		out_up3 = self.up3(out_down3) 
# 		out_up2 = self.up2(torch.add(out_up3,out_down2)) 
# 		out_up1 = self.up1(torch.add(out_up2,out_down1)) 
# 		out_up0 = self.up0(torch.add(out_up1,out_down0))

# 		return out_up0


# class dice_loss(nn.Module):
# 	def __init__(self):
# 		super(dice_loss, self).__init__()

# 	def forward(self,output,target): # (N,DHW) two Variables
# 		smooth = 1
# 		output = torch.max(output,dim=1)[1].float()
# 		intersect = torch.mul(output,target)
# 		score = 2*(intersect.sum()+smooth)/(output.sum()+target.sum()+smooth)
# 		# print(intersect.sum(1),output.sum(1),target.sum(1))
# 		score = 100*(1 - score.sum())
# 		print(score)
# 		return score

import bioloss


if __name__ == '__main__':

	model = CliqueNet(input_channels=8, list_channels=[8, 8, 8], list_layer_num=[5, 5, 5])
	if torch.cuda.is_available():
		model = torch.nn.DataParallel(model, device_ids=gpu_list).cuda()

	optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=0.0001)
	# optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30,40], gamma=0.1)
	scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=6, threshold=0.04, threshold_mode='abs')

	for e in range(epoch):

		model.train()
		total_loss = 0.0
		# scheduler.step() 

		for index,(image,target) in enumerate(trainloader):

			# print ("Train Epoch[%d/%d], Iter[%d]" %(e+1, epoch, index))			
			optimizer.zero_grad()
			image, target = to_var(image,volatile=False), to_var(target,volatile=False).float()
			output = model(image) # (N,HW)

			target = target.view(target.numel())
			loss = bioloss.dice_loss(output, target)
			#loss = F.nll_loss(output, target)
			total_loss += loss.data[0]
			loss.backward()
			optimizer.step()

			# if index == 0 and e%10 == 9:
			# 	image = image.data.cpu().numpy().reshape(-1,512,512)
			# 	target = target.data.cpu().numpy().reshape(-1,512,512)
			# 	output = output.data.max(dim=1)[1].cpu().numpy().reshape(-1,512,512)

			# 	for i in range(batch_size):
			# 		plt.figure(figsize=(100,100))
			# 		plt.subplot(1,3,1)
			# 		plt.imshow(image[i],cmap="gray") # original image
			# 		plt.subplot(1,3,2)
			# 		plt.imshow(target[i],cmap="Set1") # ground truth
			# 		plt.subplot(1,3,3)
			# 		plt.imshow(output[i],cmap="Set1") # my prediction
			# 		plt.show()


		print ("Epoch[%d/%d], Train Dice Coef: %.4f" %(e+1, epoch, total_loss/len(trainloader)))


		model.eval()
		total_loss = 0.0
		new_loss = 0.0 

		for index,(image,target) in enumerate(testloader):

			# print ("Valid Epoch[%d/%d], Iter[%d]" %(e+1, epoch, index))	
			image, target = to_var(image,volatile=True), to_var(target,volatile=True).float()
			output = model(image)

			target = target.view(target.numel()) # (NDHW)
			# total_loss += F.nll_loss(output, target)
			loss = bioloss.dice_loss(output, target)
			total_loss += loss.data[0]


			del image, target, loss, output #if e == 50 or e == 32:
			# 	image = image.data.cpu().numpy().reshape(-1,512,512)
			# 	target = target.data.cpu().numpy().reshape(-1,512,512)
			# 	output = output.data.max(dim=1)[1].cpu().numpy().reshape(-1,512,512)

			# 	if index == 0:
			# 		image_save = image 
			# 		target_save = target 
			# 		output_save = output 
			# 	elif index == 1:
			# 		image_save = np.concatenate((image_save,image),axis=0)
			# 		target_save = np.concatenate((target_save,target),axis=0)
			# 		output_save = np.concatenate((output_save,output),axis=0)
			# 	else:
			# 		image_save = np.concatenate((image_save,image),axis=0)
			# 		target_save = np.concatenate((target_save,target),axis=0)
			# 		output_save = np.concatenate((output_save,output),axis=0)
			# 		print(image_save.shape,target_save.shape,output_save.shape)

			# 		if e == 50:
			# 			np.save('data/image_save_50.npy',image_save)
			# 			np.save('data/target_save_50.npy',target_save)
			# 			np.save('data/output_save_50.npy',output_save)
			# 		else:
			# 			np.save('data/image_save_32.npy',image_save)
			# 			np.save('data/target_save_32.npy',target_save)
			# 			np.save('data/output_save_32.npy',output_save)


			# if index == 0 and e%10 == 9:
			# 	image = image.data.cpu().numpy().reshape(-1,512,512)
			# 	target = target.data.cpu().numpy().reshape(-1,512,512)
			# 	output = output.data.max(dim=1)[1].cpu().numpy().reshape(-1,512,512)

			# 	for i in range(batch_size):
			# 		plt.figure(figsize=(100,100))
			# 		plt.subplot(1,3,1)
			# 		plt.imshow(image[i],cmap="gray") # original image
			# 		plt.subplot(1,3,2)
			# 		plt.imshow(target[i],cmap="Set1") # ground truth
			# 		plt.subplot(1,3,3)
			# 		plt.imshow(output[i],cmap="Set1") # my prediction
			# 		plt.show()


		print ("Epoch[%d/%d], Valid Dice Coef: %.4f" %(e+1, epoch, total_loss/len(testloader)))
		scheduler.step(total_loss/len(testloader))
		print('learning rate',optimizer.param_groups[0]['lr'])
		# print ("Epoch[%d/%d], Valid Loss: %.2f, Valid Acc: %.2f" %(e+1, epoch, total_loss, 100*accuracy/cnt))




	# print('total time cost: %s'%(str(datetime.now()-start)[:7]))
	# torch.save(vnet.state_dict(),'vnet'+str(datetime.now())[5:16]+'.pkl')
