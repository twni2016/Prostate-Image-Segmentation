import SimpleITK as sitk
import numpy as np
from scipy.misc import imresize
import matplotlib.pyplot as plt

# root = './data/promise12/Case'
# image_set = []
# label_set = []

# for i in range(50): 
# 	if i<10 :
# 		image = sitk.ReadImage(root+'0'+str(i)+'.mhd')
# 		label = sitk.ReadImage(root+'0'+str(i)+'_segmentation.mhd')
# 	else:
# 		image = sitk.ReadImage(root+str(i)+'.mhd')
# 		label = sitk.ReadImage(root+str(i)+'_segmentation.mhd')

# 	image = sitk.GetArrayFromImage(image)
# 	label = sitk.GetArrayFromImage(label)

# 	image[image>255] = 255 # [0,255]

# 	depth = image.shape[0]
# 	height = image.shape[1]
# 	width = image.shape[2]

# 	image_3d = np.zeros((64,512,512))
# 	label_3d = np.zeros((64,512,512))

# 	image_3d[(65-depth)//2:(65+depth)//2,(513-height)//2:(513+height)//2,(513-width)//2:(513+width)//2] = image
# 	label_3d[(65-depth)//2:(65+depth)//2,(513-height)//2:(513+height)//2,(513-width)//2:(513+width)//2] = label

# 	image_set.append(image_3d)
# 	label_set.append(label_3d)

# 	print(i)

# image_set = np.asarray(image_set).reshape(-1,64,512,512)
# label_set = np.asarray(label_set).reshape(-1,64,512,512)
# print(image_set.shape,label_set.shape)

# image_set = image_set.astype(np.uint8)
# label_set = label_set.astype(np.uint8)

# np.save('data/PROS_image.npy',image_set)
# np.save('data/PROS_label.npy',label_set)



# image_set = np.load('data/PROS_image.npy') # (50,64,512,512)
# label_set = np.load('data/PROS_label.npy')

# print(image_set.dtype)

# index = np.arange(50)
# np.random.shuffle(index)

# train_image_set = []
# train_label_set = []
# test_image_set = []
# test_label_set = []


# for i in index[:44]: 
# 	train_image_set.append(image_set[i])
# 	train_image_set.append(np.flip(image_set[i],axis=0))
# 	train_image_set.append(np.flip(image_set[i],axis=1))
# 	train_image_set.append(np.flip(image_set[i],axis=2))
# 	train_image_set.append(np.roll(image_set[i],shift=8,axis=0))
# 	train_image_set.append(np.roll(image_set[i],shift=64,axis=1))
# 	train_image_set.append(np.roll(image_set[i],shift=64,axis=2))

# 	train_label_set.append(label_set[i])
# 	train_label_set.append(np.flip(label_set[i],axis=0))
# 	train_label_set.append(np.flip(label_set[i],axis=1))
# 	train_label_set.append(np.flip(label_set[i],axis=2))
# 	train_label_set.append(np.roll(label_set[i],shift=8,axis=0))
# 	train_label_set.append(np.roll(label_set[i],shift=64,axis=1))
# 	train_label_set.append(np.roll(label_set[i],shift=64,axis=2))
# 	print(i)


# for i in index[44:]: 
# 	test_image_set.append(image_set[i])
# 	test_image_set.append(np.flip(image_set[i],axis=0))
# 	test_image_set.append(np.flip(image_set[i],axis=1))
# 	test_image_set.append(np.flip(image_set[i],axis=2))
# 	test_image_set.append(np.roll(image_set[i],shift=8,axis=0))
# 	test_image_set.append(np.roll(image_set[i],shift=64,axis=1))
# 	test_image_set.append(np.roll(image_set[i],shift=64,axis=2))

# 	test_label_set.append(label_set[i])
# 	test_label_set.append(np.flip(label_set[i],axis=0))
# 	test_label_set.append(np.flip(label_set[i],axis=1))
# 	test_label_set.append(np.flip(label_set[i],axis=2))
# 	test_label_set.append(np.roll(label_set[i],shift=8,axis=0))
# 	test_label_set.append(np.roll(label_set[i],shift=64,axis=1))
# 	test_label_set.append(np.roll(label_set[i],shift=64,axis=2))
# 	print(i)

# train_image_set = np.asarray(train_image_set).reshape(-1,64,512,512)
# train_label_set = np.asarray(train_label_set).reshape(-1,64,512,512)
# test_image_set = np.asarray(test_image_set).reshape(-1,64,512,512)
# test_label_set = np.asarray(test_label_set).reshape(-1,64,512,512)

# print(train_image_set.shape,train_label_set.shape,test_image_set.shape,test_label_set.shape)

# np.save('data/PROS_train_image.npy',train_image_set)
# np.save('data/PROS_train_label.npy',train_label_set)
# np.save('data/PROS_test_image.npy',test_image_set)
# np.save('data/PROS_test_label.npy',test_label_set)

print('loading...')
train_image_set = np.load('data/PROS_train_image.npy') # (308,64,512,512)
train_label_set = np.load('data/PROS_train_label.npy')
test_image_set = np.load('data/PROS_test_image.npy') # (42,64,512,512)
test_label_set = np.load('data/PROS_test_label.npy')


import torch.utils.data as data

class PROS12(data.Dataset):
	def __init__(self, train=True, transform=None):
		self.train = train
		self.transform = transform
		# now load dataset
		if self.train is True:

			self.train_data = train_image_set 
			self.train_data = self.train_data.reshape(-1,1,64,512,512) # NCDHW
			self.train_label = train_label_set 
			self.train_label = self.train_label.reshape(-1,1,64,512,512) # NCDHW

		else:

			self.test_data = test_image_set 
			self.test_data = self.test_data.reshape(-1,1,64,512,512) # NCDHW
			self.test_label = test_label_set 
			self.test_label = self.test_label.reshape(-1,1,64,512,512) # NCDHW

	def __getitem__(self, index):
		if self.train is True:
			img, target = self.train_data[index], self.train_label[index]
		else:
			img, target = self.test_data[index], self.test_label[index]

		if self.transform is not None:
			img = self.transform(img)

		return img, target

	def __len__(self):
		if self.train is True:
			return len(self.train_data)
		else:
			return len(self.test_data)

print('dataset is loaded!')

