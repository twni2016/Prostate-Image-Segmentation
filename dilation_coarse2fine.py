# Best Model: valid dice: 0.81
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets
from torchvision import transforms
import torchvision as tv
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

	# argparse settings
	import argparse
	parser = argparse.ArgumentParser(description='PROS12')
	parser.add_argument('-b', '--batch', type=int, default=32, help='input batch size for training (default: 64)')
	parser.add_argument('-e', '--epoch', type=int, default=50, help='number of epochs to train (default: 50)')
	parser.add_argument('--lr', type=float, default=0.001, help='learning rate (default: 0.001)')
	parser.add_argument('--gpu', type=int, default=4, help='GPU (default: 4)')
	args = parser.parse_args()


	# HyperParameter
	epoch = args.epoch
	batch_size = args.batch
	lr = args.lr
	gpu_list = [item for item in range(args.gpu)]

	transform = transforms.Compose([
		transforms.ToTensor(),
		transforms.Normalize((0.5,),(0.5,)) # tuple for one channel
	])

	from dataset_coarse2fine import PROS12, FineSet
	training_set = PROS12(train=True, transform=transform)
	testing_set = PROS12(train=False,transform=transform)

	trainloader = torch.utils.data.DataLoader(training_set, batch_size=batch_size, shuffle=True)
	testloader = torch.utils.data.DataLoader(testing_set, batch_size=batch_size, shuffle=True)

	def fine_collate(batch):
		image = to_var(torch.FloatTensor(batch[0][0]))
		target = to_var(torch.FloatTensor(batch[0][1]))
		return image, target

	fine_training_set = FineSet()
	fineloader = torch.utils.data.DataLoader(fine_training_set, batch_size=1, shuffle=True, collate_fn=fine_collate)


def to_var(x, volatile=False):
	if torch.cuda.is_available():
		x = x.cuda()
	return Variable(x,volatile=volatile)


class DownTransition(nn.Module):

	def __init__(self,inchan,outchan,layer,dilation_=1):
		super(DownTransition, self).__init__()
		self.dilation_ = dilation_
		self.outchan = outchan
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

	def __init__(self,inchan,outchan,layer,last=False):
		super(UpTransition, self).__init__()
		self.last = last
		self.outchan = outchan
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
			# print('out',out.shape)
		return out 

class Vnet(nn.Module):
	# 1*512*512
	def __init__(self, inchans, outchans, down_layers, up_layers, dilations):
		super(Vnet,self).__init__()

		self.layer0 = nn.Sequential(
				nn.Conv2d(1, 8, kernel_size=7, stride=1, padding=3,bias=False),
				nn.BatchNorm2d(8,affine=True),
				nn.ELU(inplace=True)
			)

		self.block_num = len(inchans)
		self.down = nn.ModuleList() # must create pytorch module list
		self.up = nn.ModuleList()

		for i in range(self.block_num):
			self.down.append(DownTransition(inchan=inchans[i], outchan=outchans[i], layer=down_layers[i], dilation_=dilations[i]))
			if i==0 :
				self.up.append(UpTransition(inchan=outchans[i], outchan=inchans[i], layer=up_layers[i], last=True))
			else:
				self.up.append(UpTransition(inchan=outchans[i], outchan=inchans[i], layer=up_layers[i]))


		# self.down0 = DownTransition(inchan=8,outchan=32,layer=3,dilation_=2)  # 32*128^2
		# self.down1 = DownTransition(inchan=32,outchan=128,layer=3,dilation_=2) # 128*64^2
		# self.down2 = DownTransition(inchan=128,outchan=256,layer=3,dilation_=4) # 256*32^2
		# # self.down3 = DownTransition(inchan=64,outchan=128,layer=2,dilation_=4) # 128*32^2

		# # self.up3 = UpTransition(inchan=128,outchan=64,layer=2) # 64*64^2
		# self.up2 = UpTransition(inchan=256,outchan=128,layer=2) # 32*128^2
		# self.up1 = UpTransition(inchan=128,outchan=32,layer=2) # 16*256^2
		# self.up0 = UpTransition(inchan=32,outchan=8,layer=2,last=True) # 2*512^2

		for m in self.modules():
			if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
				nn.init.kaiming_normal(m.weight.data)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

	def forward(self,x):
		
		x = self.layer0(x)
		# print('init',x.shape)

		out_down = [] # must create list
		out_down.append(self.down[0](x))
		# print('0',out_down[0].shape)

		for i in range(1,self.block_num):
			out_down.append(self.down[i](out_down[i-1]))
			# print(i,out_down[i].shape)

		out_up = self.up[self.block_num-1](out_down[self.block_num-1])
		# print('2',out_up.shape)

		for i in reversed(range(self.block_num-1)):
			out_up = self.up[i](torch.add(out_up,out_down[i])) # iterate out_up
			# print(i,out_up.shape)

		return out_up

		# out_down0 = self.down0(x)        
		# out_down1 = self.down1(out_down0)
		# out_down2 = self.down2(out_down1) 
		# # out_down3 = self.down3(out_down2) 

		# out_up2 = self.up2(out_down2) 
		# # out_up2 = self.up2(torch.add(out_up3,out_down2)) 
		# out_up1 = self.up1(torch.add(out_up2,out_down1)) 
		# out_up0 = self.up0(torch.add(out_up1,out_down0))

		# return out_up0


import bioloss


if __name__ == '__main__':

	coarse_vnet = Vnet(inchans=[8,32,128], outchans=[32,128,256], down_layers=[3,3,3], up_layers=[2,2,2], dilations=[2,2,4])
	fine_vnet = Vnet(inchans=[8,32,128], outchans=[32,128,256], down_layers=[3,3,3], up_layers=[2,2,2], dilations=[2,2,4])

	if torch.cuda.is_available():
		coarse_vnet = torch.nn.DataParallel(coarse_vnet, device_ids=gpu_list).cuda()
		fine_vnet = torch.nn.DataParallel(fine_vnet, device_ids=gpu_list).cuda()

	coarse_optimizer = torch.optim.Adam(coarse_vnet.parameters(), lr=lr, weight_decay=0.0001)
	fine_optimizer = torch.optim.Adam(fine_vnet.parameters(), lr=lr, weight_decay=0.0001)
	# optimizer = torch.optim.SGD(coarse_vnet.parameters(), lr=lr, momentum=0.9, weight_decay=0.0001, nesterov=True)
	# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[14,24,34], gamma=0.1)
	coarse_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(coarse_optimizer, mode='max', patience=6, threshold=0.04, threshold_mode='abs')
	fine_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(fine_optimizer, mode='max', patience=6, threshold=0.04, threshold_mode='abs')

	for e in range(epoch):

		coarse_vnet.train()
		total_coarse_loss = 0.0
		# scheduler.step() 

		for index,(image,target) in enumerate(trainloader):

			# print ("Train Epoch[%d/%d], Iter[%d]" %(e+1, epoch, index))           
			coarse_optimizer.zero_grad()
			image, target = to_var(image,volatile=False), to_var(target,volatile=False).float()
			output = coarse_vnet(image) # (N,HW)
			target = target.view(target.numel())
			coarse_loss = bioloss.dice_loss(output, target)
			total_coarse_loss += coarse_loss.data[0]
			coarse_loss.backward()
			coarse_optimizer.step()

		print ("Epoch[%d/%d], Train Dice Coef Coarse Model: %.4f" %(e+1, epoch, total_coarse_loss/len(trainloader)))

			# image_center = image[:,:,144:-144,144:-144] #(N,C,100,-100)
			# target_center = target[:,144:-144,144:-144,:]
			# output = coarse_vnet(image_center) # (NHW,2)

			# target_center = target_center.contiguous().view(target_center.numel())
			# loss = bioloss.dice_loss(output, target_center)
			#loss = F.nll_loss(output, target)

			# true_output = to_var(torch.zeros(image.shape[0],512,512,2),volatile=False) # softmax
			# true_output[:,:,:,0] = 1
			# true_output[:,144:-144,144:-144] = output.contiguous().view(image.shape[0],224,224,2)
			# true_output = true_output.contiguous().view(-1,2)
			# target = target.contiguous().view(target.numel())
			# true_loss += bioloss.dice_loss(true_output,target).data[0]


			# if index == 0 and e%10 == 9:
			#   image = image.data.cpu().numpy().reshape(-1,512,512)
			#   target = target.data.cpu().numpy().reshape(-1,512,512)
			#   output = output.data.max(dim=1)[1].cpu().numpy().reshape(-1,512,512)

			#   for i in range(batch_size):
			#       plt.figure(figsize=(100,100))
			#       plt.subplot(1,3,1)
			#       plt.imshow(image[i],cmap="gray") # original image
			#       plt.subplot(1,3,2)
			#       plt.imshow(target[i],cmap="Set1") # ground truth
			#       plt.subplot(1,3,3)
			#       plt.imshow(output[i],cmap="Set1") # my prediction
			#       plt.show()

		if e<17:
			continue

		fine_vnet.train()
		total_fine_loss = 0.0
		total_loss = 0.0
		real_batch = 30 # 3390

		for index,(image,target) in enumerate(fineloader):

			# print ("Train Epoch[%d/%d], Iter[%d]" %(e+1, epoch, index))           
			# image, target = to_var(image,volatile=False), to_var(target,volatile=False).float()
			output = fine_vnet(image) # (N,HW)
			target = target.view(target.numel())
			fine_loss = bioloss.dice_loss(output, target)
			total_fine_loss += fine_loss
			total_loss += fine_loss.data[0]

			if (index+1)%real_batch == 0:
				fine_optimizer.zero_grad()
				ave_loss = total_fine_loss/real_batch
				ave_loss.backward()
				fine_optimizer.step()
				total_fine_loss = 0

		print ("Epoch[%d/%d], Train Dice Coef Fine Model: %.4f" %(e+1, epoch, total_loss/len(fineloader)))


		if e<20:
			continue

		coarse_vnet.eval()
		fine_vnet.eval()
		total_coarse_loss = 0.0
		total_fine_loss = 0.0

		for index,(image,target) in enumerate(testloader):
 			
			# print('coarse stage')
			image, target = to_var(image,volatile=True), to_var(target,volatile=True).float()
			output = coarse_vnet(image)
			# print('output',output.shape)
			target = target.view(target.numel())
			total_coarse_loss += bioloss.dice_loss(output, target).data[0]

			predict_map = (output.max(1)[1]).data.float().view(batch_size,512,512)
			# print(type(predict_map))		
			margin = 16

			# print('fine stage')
			for i in range(batch_size):
				print(i)
				origin_output = output.contiguous().view(batch_size,-1,2)[i]
				origin_output = origin_output.contiguous().view(-1,2)
				true_target = target.contiguous().view(batch_size,-1)[i]
				true_target = true_target.contiguous().view(true_target.numel())
				origin_loss = bioloss.dice_loss(origin_output,true_target).data[0] # fix 0/0=0 bug

				if predict_map[i].sum()==0:
					print('coarse predict none, the same in coarse and fine')
					total_fine_loss += bioloss.dice_loss(origin_output,true_target).data[0]
					continue

				h_left, h_right, v_up, v_down = (511, 0, 511, 0)

				for j in range(512):
					if predict_map[i,j].sum()>0:
						v_up = j
						break
				for j in reversed(range(512)):
					if predict_map[i,j].sum()>0:
						v_down = j+1 # [start,end)
						break
				for j in range(512):
					if predict_map[i,:,j].sum()>0:
						h_left = j
						break
				for j in reversed(range(512)):
					if predict_map[i,:,j].sum()>0:
						h_right = j+1 # [start,end)
						break

				v_remainder = 8-(v_down-v_up)%8
				h_remainder = 8-(h_right-h_left)%8
				top_margin = margin + v_remainder//2
				bottom_margin = margin + (v_remainder+1)//2
				left_margin = margin + h_remainder//2
				right_margin = margin + (h_remainder+1)//2	

				v = v_down - v_up + top_margin + bottom_margin
				h = h_right - h_left + left_margin + right_margin
				# print('v_up,v_down,h_left,h_right,top_margin,bottom_margin,left_margin,right_margin,v,h',v_up,v_down,h_left,h_right,top_margin,bottom_margin,left_margin,right_margin,v,h)

				# true_v_up = v_up-top_margin
				# true_v_down = v_down+bottom_margin
				# true_h_left = h_left-left_margin
				# true_h_right = h_right+right_margin

				# if true_v_up<0:

				# 	true_v_up += 8
				# 	if true_v_up<0:
				# 		true_v_up += 8
				# 	if true_v_up<0:
				# 		true_v_up = v_up
				# 	true_v_down +=

				# 	while true_v_up<0 or true_v_up>v_up:
				# 		true_v_up += 8
				# 	if true_v_up>v_up:
				# 		true_v_up = v_up
				# 		true_v_down += 




				uni_image = image.data[i,0,(v_up-top_margin):(v_down+bottom_margin),(h_left-left_margin):(h_right+right_margin)] # NCHW
				uni_image = to_var(uni_image.contiguous().view(1,1,v,h),volatile=True)
				print('uni_image.shape',uni_image.shape)

				fine_output = fine_vnet(uni_image)
				# print('fine_output.shape,v*h',fine_output.shape,v*h)
				true_output = to_var(torch.zeros(512,512,2),volatile=True) # softmax
				true_output[:,:,0] = 1
				true_output[(v_up-top_margin):(v_down+bottom_margin),(h_left-left_margin):(h_right+right_margin)] = fine_output.contiguous().view(v,h,2)
				true_output = true_output.contiguous().view(-1,2)
				print('coarse predict exist, comparsion between coarse and fine')
				total_fine_loss += bioloss.dice_loss(true_output,true_target).data[0]

				# 严重问题：fine model预测偏向都是1，导致结果反而更差



			# image_center = image[:,:,144:-144,144:-144] #(N,C,100,-100)
			# target_center = target[:,144:-144,144:-144,:]
			# output = coarse_vnet(image_center) # (NHW,2)

			# target_center = target_center.contiguous().view(target_center.numel())
			# loss = bioloss.dice_loss(output, target_center)
			# #loss = F.nll_loss(output, target)
			# total_loss += loss.data[0]

			# true_output = to_var(torch.zeros(image.shape[0],512,512,2),volatile=True) # softmax
			# true_output[:,:,:,0] = 1
			# true_output[:,144:-144,144:-144] = output.contiguous().view(image.shape[0],224,224,2)
			# true_output = true_output.contiguous().view(-1,2)
			# target = target.contiguous().view(target.numel())
			# true_loss += bioloss.dice_loss(true_output,target).data[0]

			# del image, image_center, target, target_center, output, true_output


			# if e == 50 or e == 32:
			#   image = image.data.cpu().numpy().reshape(-1,512,512)
			#   target = target.data.cpu().numpy().reshape(-1,512,512)
			#   output = output.data.max(dim=1)[1].cpu().numpy().reshape(-1,512,512)

			#   if index == 0:
			#       image_save = image 
			#       target_save = target 
			#       output_save = output 
			#   elif index == 1:
			#       image_save = np.concatenate((image_save,image),axis=0)
			#       target_save = np.concatenate((target_save,target),axis=0)
			#       output_save = np.concatenate((output_save,output),axis=0)
			#   else:
			#       image_save = np.concatenate((image_save,image),axis=0)
			#       target_save = np.concatenate((target_save,target),axis=0)
			#       output_save = np.concatenate((output_save,output),axis=0)
			#       print(image_save.shape,target_save.shape,output_save.shape)

			#       if e == 50:
			#           np.save('data/image_save_50.npy',image_save)
			#           np.save('data/target_save_50.npy',target_save)
			#           np.save('data/output_save_50.npy',output_save)
			#       else:
			#           np.save('data/image_save_32.npy',image_save)
			#           np.save('data/target_save_32.npy',target_save)
			#           np.save('data/output_save_32.npy',output_save)


			# if index == 0 and e%10 == 9:
			#   image = image.data.cpu().numpy().reshape(-1,512,512)
			#   target = target.data.cpu().numpy().reshape(-1,512,512)
			#   output = output.data.max(dim=1)[1].cpu().numpy().reshape(-1,512,512)

			#   for i in range(batch_size):
			#       plt.figure(figsize=(100,100))
			#       plt.subplot(1,3,1)
			#       plt.imshow(image[i],cmap="gray") # original image
			#       plt.subplot(1,3,2)
			#       plt.imshow(target[i],cmap="Set1") # ground truth
			#       plt.subplot(1,3,3)
			#       plt.imshow(output[i],cmap="Set1") # my prediction
			#       plt.show()

		print ("Epoch[%d/%d], Valid Dice Coef Coarse Model: %.4f" %(e+1, epoch, total_coarse_loss/len(testloader)))
		print ("Epoch[%d/%d], Valid Dice Coef Fine Model: %.4f" %(e+1, epoch, total_fine_loss/192))
		coarse_scheduler.step(total_fine_loss/192)
		fine_scheduler.step(total_fine_loss/192)
		print('learning rate',coarse_optimizer.param_groups[0]['lr'],fine_optimizer.param_groups[0]['lr'])

		# print ("Epoch[%d/%d], Valid Loss: %.2f, Valid Acc: %.2f" %(e+1, epoch, total_loss, 100*accuracy/cnt))




	# print('total time cost: %s'%(str(datetime.now()-start)[:7]))
	# torch.save(coarse_vnet.state_dict(),'coarse_vnet'+str(datetime.now())[5:16]+'.pkl')
