# PROMISE 12
# 50 3D pics with 1377 transveral pics. max 512*512 min 256*256
# padding to 512*512
# training set: 0-43; test 44-49. 1233/1377=89.5%

# import SimpleITK as sitk
# from scipy import ndimage
import numpy as np 
import torch.utils.data as data


# root = './data/PROS_Train/Case'
# image_set = np.array([],dtype=np.int16).reshape(0,512,512) # actually 16 bits
# label_set = np.array([],dtype=np.int8).reshape(0,512,512) # actually 1 bit

# # first shuffle 
# index = np.arange(50)
# np.random.shuffle(index)

# for i in index: 
# 	if i<10 :
# 		image = sitk.ReadImage(root+'0'+str(i)+'.mhd')
# 		label = sitk.ReadImage(root+'0'+str(i)+'_segmentation.mhd')
# 	else:
# 		image = sitk.ReadImage(root+str(i)+'.mhd')
# 		label = sitk.ReadImage(root+str(i)+'_segmentation.mhd')

# 	image = sitk.GetArrayFromImage(image)
# 	label = sitk.GetArrayFromImage(label)

# 	depth = image.shape[0]
# 	height = image.shape[1]
# 	width = image.shape[2]

# 	image_2d = np.zeros((depth,512,512))
# 	label_2d = np.zeros((depth,512,512))

# 	image_2d[:,(513-height)//2:(513+height)//2,(513-width)//2:(513+width)//2] = image
# 	label_2d[:,(513-height)//2:(513+height)//2,(513-width)//2:(513+width)//2] = label
# 	print(image_2d.shape)

# 	image_set = np.concatenate((image_set,image_2d),axis=0)
# 	label_set = np.concatenate((label_set,label_2d),axis=0) # debug...

# 	print(i)

# print(image_set.shape,label_set.shape,image_set.dtype,label_set.dtype)
# print('image_set:max/min',image_set.max(),image_set.min())
# print('label_set:max/min',label_set.max(),label_set.min())

# image_set = image_set.astype(np.int16)
# label_set = label_set.astype(np.int8)
# print(image_set.shape,label_set.shape,image_set.dtype,label_set.dtype)
# print('image_set:max/min',image_set.max(),image_set.min())
# print('label_set:max/min',label_set.max(),label_set.min())

# np.save('data/image2d.npy',image_set)
# np.save('data/label2d.npy',label_set)



# 2018.1.15 以上

print('loading')
image_set = np.load('data/image2d.npy')
label_set = np.load('data/label2d.npy')
print('finish loading')

train_image_set = np.zeros((1185*5,512,512))
train_label_set = np.zeros((1185*5,512,512))
train_image_set[:1185] = image_set[:1185] # 192 test images
train_label_set[:1185] = label_set[:1185]

test_image_set = image_set[1185:]
test_label_set = label_set[1185:]
print('start augmentation')

cnt = 1185
for i in range(0,1185):
	# train_image_set[cnt]   = np.flip(train_image_set[i],axis=0) # vertical flip
	# train_label_set[cnt]   = train_label_set[i]
	# train_image_set[cnt+1] = np.flip(train_image_set[i],axis=1) # horizontal flip
	# train_label_set[cnt+1] = train_label_set[i]
	train_image_set[cnt] = np.roll(train_image_set[i],shift=4,axis=0) # vertical translation
	train_label_set[cnt] = train_label_set[i]
	train_image_set[cnt+1] = np.roll(train_image_set[i],shift=-4,axis=0) # vertical translation
	train_label_set[cnt+1] = train_label_set[i]
	train_image_set[cnt+2] = np.roll(train_image_set[i],shift=4,axis=1) # horizontal translation
	train_label_set[cnt+2] = train_label_set[i]
	train_image_set[cnt+3] = np.roll(train_image_set[i],shift=-4,axis=1) # horizontal translation
	train_label_set[cnt+3] = train_label_set[i]

	cnt += 4

print('end augmentation')

train_image_set = train_image_set.astype(np.float32) # PyTorch default floarTensor is 32 bit
train_label_set = train_label_set.astype(np.float32)
test_image_set = test_image_set.astype(np.float32)
test_label_set = test_label_set.astype(np.float32)

class PROS12(data.Dataset):
	def __init__(self, train=True, transform=None):
		self.train = train
		self.transform = transform
		# now load dataset
		if self.train is True:

			self.train_data = train_image_set 
			self.train_data = self.train_data.reshape(-1,1,512,512) # NCHW
			self.train_data = self.train_data.transpose((0, 2, 3, 1)) # NHWC
			self.train_label = train_label_set 
			self.train_label = self.train_label.reshape(-1,1,512,512) # NCHW
			self.train_label = self.train_label.transpose((0, 2, 3, 1)) 

		else:

			self.test_data = test_image_set 
			self.test_data = self.test_data.reshape(-1,1,512,512) # NCHW
			self.test_data = self.test_data.transpose((0, 2, 3, 1)) 
			self.test_label = test_label_set 
			self.test_label = self.test_label.reshape(-1,1,512,512) # NCHW
			self.test_label = self.test_label.transpose((0, 2, 3, 1)) 

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

print('dataset is entirely loaded!')







