# Baseline Code
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision as tv

import pydensecrf.densecrf as dcrf
from pydensecrf.utils import compute_unary, create_pairwise_bilateral, create_pairwise_gaussian, unary_from_softmax

if __name__ == '__main__':

	# argparse settings
	import argparse
	parser = argparse.ArgumentParser(description='PROS12')
	parser.add_argument('-b', '--batch', type=int, default=64, help='input batch size for training (default: 64)')
	parser.add_argument('-e', '--epoch', type=int, default=75, help='number of epochs to train (default: 50)')
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
	parser.add_argument('--gpu', type=int, default=4, help='GPU (default: 4)')
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


def to_var(x):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x)


class DownTransition(nn.Module):

	def __init__(self,inchan,layer,dilation_=1):
		super(DownTransition, self).__init__()
		self.dilation_ = dilation_
		if inchan == 1: 
			self.outchan = 8
		else:
			self.outchan = 2*inchan
		self.layer = layer
		self.down = nn.Conv2d(in_channels=inchan,out_channels=self.outchan,kernel_size=3,padding=1,stride=2) # /2
		self.bn = nn.BatchNorm2d(num_features=self.outchan,affine=True)
		self.conv = self.make_layers()
		self.relu = nn.ELU(inplace=True)

	def make_layers(self):

		layers = []
		for i in range(self.layer):
			layers.append(nn.ELU(inplace=True))
			# padding = dilation
			layers.append(nn.Conv2d(self.outchan,self.outchan,kernel_size=3,padding=self.dilation_,stride=1,dilation=self.dilation_))
			layers.append(nn.BatchNorm2d(num_features=self.outchan,affine=True))
			
		return nn.Sequential(*layers)

	def forward(self,x):

		out1 = self.down(x)
		out2 = self.conv(self.bn(out1))
		out2 = self.relu(torch.add(out1,out2))
		return out2

class UpTransition(nn.Module):

	def __init__(self,inchan,layer,last=False):
		super(UpTransition, self).__init__()
		self.last = last
		self.outchan = inchan//2
		self.layer = layer
		self.up =  nn.ConvTranspose2d(in_channels=inchan,out_channels=self.outchan,kernel_size=4,padding=1,stride=2) # *2
		self.bn = nn.BatchNorm2d(num_features=self.outchan,affine=True)
		self.conv = self.make_layers()
		self.relu = nn.ELU(inplace=True)
		if self.last is True:
			self.conv1 = nn.Conv2d(self.outchan,2,kernel_size=1) # 1*1 conv
			self.softmax = F.softmax

	def make_layers(self):

		layers = []
		for i in range(self.layer):
			layers.append(nn.ELU(inplace=True))
			layers.append(nn.Conv2d(self.outchan,self.outchan,kernel_size=3,padding=1,stride=1))
			layers.append(nn.BatchNorm2d(num_features=self.outchan,affine=True))
			
		return nn.Sequential(*layers)

	def forward(self,x):

		out1 = self.up(x)
		out = self.conv(self.bn(out1))
		out = self.relu(torch.add(out1,out))
		if self.last is True:
			out = self.conv1(out)
			out = out.permute(0, 2, 3, 1).contiguous()
			# print('forward',out.shape)
			# flatten to (N,HW,C=2)
			# out = out.view(out.size(0),-1,2)
			# out = self.softmax(out,dim=2)
			# out = torch.max(out,dim=2)[1].float()
			# print('softmax',out.shape)
			# result (N,HW)
			out = out.view(out.numel() // 2, 2)
			out = self.softmax(out,dim=1) # default
		return out 

class Vnet(nn.Module):
	# 1*512*512
	def __init__(self):
		super(Vnet,self).__init__()

		self.layer0 = nn.Sequential(
				nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=3,bias=False),
				nn.BatchNorm2d(8,affine=True),
				nn.ELU(inplace=True)
			)
		self.down0 = DownTransition(inchan=8,layer=2,dilation_=2)  # 16*256^2
		self.down1 = DownTransition(inchan=16,layer=2,dilation_=2) # 32*128^2
		self.down2 = DownTransition(inchan=32,layer=2,dilation_=2) # 64*64^2
		self.down3 = DownTransition(inchan=64,layer=2,dilation_=4) # 128*32^2
		

		self.up3 = UpTransition(inchan=128,layer=2) # 64*64^2
		self.up2 = UpTransition(inchan=64,layer=2) # 32*128^2
		self.up1 = UpTransition(inchan=32,layer=2) # 16*256^2
		self.up0 = UpTransition(inchan=16,layer=2,last=True) # 2*512^2

		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				nn.init.kaiming_normal(m.weight.data)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self,x):
		
		x = self.layer0(x)
		out_down0 = self.down0(x)        
		out_down1 = self.down1(out_down0)
		out_down2 = self.down2(out_down1) 
		out_down3 = self.down3(out_down2) 

		out_up3 = self.up3(out_down3) 
		out_up2 = self.up2(torch.add(out_up3,out_down2)) 
		out_up1 = self.up1(torch.add(out_up2,out_down1)) 
		out_up0 = self.up0(torch.add(out_up1,out_down0))

		return out_up0


class dice_loss(nn.Module):
	def __init__(self):
		super(dice_loss, self).__init__()

	def forward(self,output,target): # (N,DHW) two Variables
		smooth = 1
		output = torch.max(output,dim=1)[1].float()
		intersect = torch.mul(output,target)
		score = 2*(intersect.sum()+smooth)/(output.sum()+target.sum()+smooth)
		# print(intersect.sum(1),output.sum(1),target.sum(1))
		score = 100*(1 - score.sum())
		print(score)
		return score

import bioloss

if __name__ == '__main__':

	vnet = Vnet()
	if torch.cuda.is_available():
		vnet = torch.nn.DataParallel(vnet, device_ids=gpu_list).cuda()

	optimizer = torch.optim.Adam(vnet.parameters(), lr=lr, weight_decay=0.0001)
	# optimizer = torch.optim.SGD(vnet.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30,40], gamma=0.1)

	criterion = dice_loss()

	for e in range(epoch):

		vnet.train()
		total_loss = 0.0
		scheduler.step() 

		for index,(image,target) in enumerate(trainloader):

			# print ("Train Epoch[%d/%d], Iter[%d]" %(e+1, epoch, index))			
			optimizer.zero_grad()
			image, target = to_var(image), to_var(target).float()
			output = vnet(image) # (N,HW)

			target = target.view(target.numel())
			loss = bioloss.dice_loss(output, target)
			#loss = F.nll_loss(output, target)
			total_loss += loss
			loss.backward()
			optimizer.step()

			# del loss
			# del output

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


		vnet.eval()
		total_loss = 0.0
		crf_loss = 0.0

		for index,(image,target) in enumerate(testloader):

			# print ("Valid Epoch[%d/%d], Iter[%d]" %(e+1, epoch, index))	
			image, target = to_var(image), to_var(target).float()
			output = vnet(image)
			# print('output',output.shape)
			target = target.view(target.numel()) # (NDHW)
			# total_loss += F.nll_loss(output, target)
			loss = bioloss.dice_loss(output, target)
			total_loss += loss

			image = image.data.cpu().numpy().reshape(-1,512,512,1) # (64,512,512,1)
			output = output.data.cpu().numpy().reshape(-1,512,512,2) # (64,512,512,2) softmax

			# print('image',image.shape,'output',output.shape)

			output = (-np.log(output)).transpose((0, 3, 1, 2))
			new_output = output

			for i in range(batch_size):

				unary = unary_from_softmax(output[i]) # ? do we need log?
				unary = np.ascontiguousarray(unary)

				d = dcrf.DenseCRF(512*512, 2)
				d.setUnaryEnergy(unary)

				feats = create_pairwise_gaussian(sdims=(10, 10), shape=(512,512))
				d.addPairwiseEnergy(feats, compat=3,
				                    kernel=dcrf.DIAG_KERNEL,
				                    normalization=dcrf.NORMALIZE_SYMMETRIC)

				feats = create_pairwise_bilateral(sdims=(50, 50), schan=(20,),
				                                   img=image[i], chdim=2)

				d.addPairwiseEnergy(feats, compat=10,
				                     kernel=dcrf.DIAG_KERNEL,
				                     normalization=dcrf.NORMALIZE_SYMMETRIC)

				Q = d.inference(10)
				res = np.array(Q).transpose(1,0)
				# print('res', res.shape)
				
				if i==0: 
					new_output = res 
				else:
					new_output = np.concatenate((new_output,res),axis=0)

			# print('new_output', new_output.shape)
			new_output = to_var(torch.from_numpy(new_output))
			loss = bioloss.dice_loss(new_output, target)
			crf_loss += loss


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
		print ("Epoch[%d/%d], Valid Dice Coef using CRF: %.4f" %(e+1, epoch, crf_loss/len(testloader)))
		# print ("Epoch[%d/%d], Valid Loss: %.2f, Valid Acc: %.2f" %(e+1, epoch, total_loss, 100*accuracy/cnt))




	# print('total time cost: %s'%(str(datetime.now()-start)[:7]))
	# torch.save(vnet.state_dict(),'vnet'+str(datetime.now())[5:16]+'.pkl')
