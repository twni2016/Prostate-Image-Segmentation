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

torch.backends.cudnn.benchmark=True

if __name__ == '__main__':

	# argparse settings
	import argparse
	parser = argparse.ArgumentParser(description='PROS12')
	parser.add_argument('-b', '--batch', type=int, default=4, help='input batch size for training (default: 64)')
	parser.add_argument('-e', '--epoch', type=int, default=30, help='number of epochs to train (default: 50)')
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
	parser.add_argument('--gpu', type=int, default=4, help='GPU (default: 4)')
	args = parser.parse_args()


	# HyperParameter
	epoch = args.epoch
	batch_size = args.batch
	lr = args.lr
	gpu_list = [item for item in range(args.gpu)]


	class myTensor(object): 
	# (1,64,320,320)
		# convert numpy array to tensor [0,1]
		# (C x D x H x W)
		def __call__(self, pic):
			# handle numpy array
			img = torch.from_numpy(pic)
			return img.float().div(255)

	from dataset3d import PROS12
	training_set = PROS12(train=True, transform=myTensor())
	testing_set = PROS12(train=False,transform=myTensor())

	trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
	testloader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=True)


def to_var(x, volatile):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x,volatile=volatile)


class DownTransition(nn.Module):

	def __init__(self,inchan,outchan,layer,dilation_=1):
		super(DownTransition, self).__init__()
		self.dilation_ = dilation_
		self.outchan = outchan
		self.layer = layer
		self.down = nn.Conv3d(in_channels=inchan,out_channels=self.outchan,kernel_size=3,padding=1,stride=2) # /2
		self.bn = nn.BatchNorm3d(num_features=self.outchan,affine=True)
		self.conv = self.make_layers()
		self.relu = nn.ELU(inplace=True)

	def make_layers(self):

		layers = []
		for i in range(self.layer):
			layers.append(nn.ELU(inplace=True))
			# padding = dilation
			layers.append(nn.Conv3d(self.outchan,self.outchan,kernel_size=3,padding=self.dilation_,stride=1,dilation=self.dilation_))
			layers.append(nn.BatchNorm3d(num_features=self.outchan,affine=True))
			
		return nn.Sequential(*layers)

	def forward(self,x):

		out1 = self.down(x)
		out2 = self.conv(self.bn(out1))
		out2 = self.relu(torch.add(out1,out2))
		return out2

class UpTransition(nn.Module):

	def __init__(self,inchan,outchan,layer,last=False):
		super(UpTransition, self).__init__()
		self.last = last
		self.outchan = outchan
		self.layer = layer
		self.up =  nn.ConvTranspose3d(in_channels=inchan,out_channels=self.outchan,kernel_size=4,padding=1,stride=2) # *2
		self.bn = nn.BatchNorm3d(num_features=self.outchan,affine=True)
		self.conv = self.make_layers()
		self.relu = nn.ELU(inplace=True)
		if self.last is True:
			self.conv1 = nn.Conv3d(self.outchan,2,kernel_size=1) # 1*1 conv
			self.softmax = F.softmax

	def make_layers(self):

		layers = []
		for i in range(self.layer):
			layers.append(nn.ELU(inplace=True))
			layers.append(nn.Conv3d(self.outchan,self.outchan,kernel_size=3,padding=1,stride=1))
			layers.append(nn.BatchNorm3d(num_features=self.outchan,affine=True))
			
		return nn.Sequential(*layers)

	def forward(self,x):

		out1 = self.up(x)
		out = self.conv(self.bn(out1))
		out = self.relu(torch.add(out1,out))
		if self.last is True:
			out = self.conv1(out)
			out = out.permute(0, 2, 3, 4, 1).contiguous()
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
				nn.Conv3d(1, 8, kernel_size=7, stride=1, padding=3,bias=False),
				nn.BatchNorm3d(8,affine=True),
				nn.ELU(inplace=True)
			) # 8*64*512^2
		self.down0 = DownTransition(inchan=8,outchan=16,layer=2,dilation_=2)  # 16*32*256^2
		self.down1 = DownTransition(inchan=16,outchan=32,layer=2,dilation_=2) # 32*16*128^2
		self.down2 = DownTransition(inchan=32,outchan=64,layer=2,dilation_=4) # 64*8*64^2
		self.down3 = DownTransition(inchan=64,outchan=128,layer=2,dilation_=4) # 128*4*32^2
		

		self.up3 = UpTransition(inchan=128,outchan=64,layer=2) # 64*8*64^2
		self.up2 = UpTransition(inchan=64,outchan=32,layer=2) # 32*16*128^2
		self.up1 = UpTransition(inchan=32,outchan=16,layer=2) # 16*32*256^2
		self.up0 = UpTransition(inchan=16,outchan=4,layer=2,last=True) # 2*64*512^2

		for m in self.modules():
			if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d):
				nn.init.kaiming_normal(m.weight.data)
			elif isinstance(m, nn.BatchNorm3d):
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



import bioloss


if __name__ == '__main__':

	vnet = Vnet()
	if torch.cuda.is_available():
		vnet = torch.nn.DataParallel(vnet, device_ids=gpu_list).cuda()

	optimizer = torch.optim.Adam(vnet.parameters(), lr=lr, weight_decay=0.0001)
	# optimizer = torch.optim.SGD(vnet.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
	scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20,30,40], gamma=0.1)

	for e in range(epoch):

		vnet.train()
		total_loss = 0.0
		scheduler.step() 

		for index,(image,target) in enumerate(trainloader):

			# print ("Train Epoch[%d/%d], Iter[%d]" %(e+1, epoch, index))			
			optimizer.zero_grad()
			image, target = to_var(image,volatile=False), to_var(target,volatile=False).float()
			output = vnet(image) # (NDHW,2)

			target = target.view(target.numel()) # (NDHW)
			loss = bioloss.dice_loss(output, target)
			#loss = F.nll_loss(output, target)
			total_loss += loss.data[0]
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
		new_loss = 0.0 

		for index,(image,target) in enumerate(testloader):

			# print ("Valid Epoch[%d/%d], Iter[%d]" %(e+1, epoch, index))	
			image, target = to_var(image,volatile=True), to_var(target,volatile=True).float()
			output = vnet(image) # (NDHW,2)

			target = target.view(target.numel()) # (NDHW)
			# total_loss += F.nll_loss(output, target)
			loss = bioloss.dice_loss(output, target)
			total_loss += loss.data[0]

			del image, target, loss, output


			# if e == 50 or e == 32:
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
		# print ("Epoch[%d/%d], Valid Loss: %.2f, Valid Acc: %.2f" %(e+1, epoch, total_loss, 100*accuracy/cnt))




	# print('total time cost: %s'%(str(datetime.now()-start)[:7]))
	# torch.save(vnet.state_dict(),'vnet'+str(datetime.now())[5:16]+'.pkl')
