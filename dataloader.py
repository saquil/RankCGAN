from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data.dataset import Dataset
from torch.utils.serialization import load_lua
import torchvision.transforms.functional as F
import torch


# load the dataset of images as global to avoid multiple allocations
shoe_images = None

class Shoes(Dataset):
	def __init__(self, path, transform=None):
		self.transforms = transform
		self.path = path
		global shoe_images
		if shoe_images == None:
			 shoe_images = load_lua(path)
		self.data =  shoe_images
		self.data_len = len(self.data)
		

	def __getitem__(self, index):
		img = self.data[index]
		img_tensor = img #self.transforms(F.to_pil_image(img))
		return img_tensor


	def __len__(self):
		return self.data_len

class Shoe_pairs(Dataset):
	def __init__(self, path, attribute, transform=None):
		self.transforms = transform
		self.path = path
		self.attribute = attribute
		global shoe_images
		if  shoe_images == None:
			print('no images are loaded!!')
			return
		
		if self.attribute == 'sporty':
			self.data = load_lua(path)[0][2] # select training pairs '0' with attribute '2' sporty, check Zappos dataset spec
			self.data_len = len(self.data)   # number of pairs
		elif self.attribute == 'black':
			self.data = load_lua(path) 		 # load comparison pairs
			self.data_len = len(self.data)   # number of pairs		

	def __getitem__(self, index):
		pairs = self.data[index]
		img1_tensor = shoe_images[int(pairs[0])-1] #self.transforms(F.to_pil_image(shoe_images[int(pairs[0])-1]))
		img2_tensor = shoe_images[int(pairs[1])-1] #self.transforms(F.to_pil_image(shoe_images[int(pairs[1])-1]))
		
		if self.attribute == 'sporty':
			if pairs[3] == 1: # the 3rd index is the comparison label, check Zappos dataset spec
				label = 1
			else:
				label = 0
		elif self.attribute == 'black':
			if pairs[2] == 1:
				label = 1
			else:
				label = 0

		return img1_tensor, img2_tensor, label


	def __len__(self):
		return self.data_len	


def dataloader(dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if dataset == 'shoes':
        data_loader = DataLoader(Shoes('data/images.t7', transform=transform), batch_size=batch_size, shuffle=True)

    return data_loader


def pairloader(dataset, attribute, input_size, batch_size, split='train'):
	transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
	if dataset == 'shoes':
		if attribute == 'sporty':
			data_loader = DataLoader(Shoe_pairs('data/shoe_rank.t7', attribute, transform=transform), batch_size=batch_size, shuffle=True)
		elif attribute == 'black':
			data_loader = DataLoader(Shoe_pairs('data/rank_t.t7', attribute, transform=transform), batch_size=batch_size, shuffle=True)

	return data_loader

