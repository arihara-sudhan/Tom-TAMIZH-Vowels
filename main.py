import pygame
import cv2
import torch
from torch.utils.data import DataLoader
import sys
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch.nn as nn
import faiss
import pickle

imgs = [
    "./images/12.jpg", "./images/1.jpg",
    "./images/3.jpg", "./images/2.jpg",
    "./images/4.jpg", "./images/5.jpg",
    "./images/7.jpg","./images/6.jpg",
    "./images/8.jpg","./images/9.jpg",
    "./images/11.jpg","./images/10.jpg",
]

x = [np.array(Image.open(img)) for img in imgs]  # Open image without conversion
y = [Image.open(img) for img in imgs]  # Open image without conversion

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

labelsd = {'0':"அ",
           '1':'ஆ',
           '2':'இ',
           '3':'ஈ',
          '4':'உ',
           '5':'ஊ',
           '6':'எ',
           '7':'ஏ',
           '8':'ஐ',
           '9':'ஒ',
           '10':'ஓ',
           '11':'ஔ'
          }

with open('embeddings/embedding_data.pkl', 'rb') as f:
    data = pickle.load(f)

embs = data['embs']
labels = data['labels']

nlist = 100  # Number of cells/buckets
quantizer = faiss.IndexFlatL2(embs.shape[1])  # Quantizer index (same as IndexFlatL2)
index = faiss.IndexIVFFlat(quantizer, embs.shape[1], nlist)
index.train(embs)
index.add(embs)
 
w,h = 1600,900
white = (255,255,255)



class EmbeddingNet(nn.Module):
    def __init__(self):
        super(EmbeddingNet, self).__init__()
        self.convnet = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(32, 64, 5),
            nn.PReLU(),
            nn.MaxPool2d(2, stride=2)
        )

        self.fc = nn.Sequential(
            nn.Linear(64 * 53 * 53, 256),
            nn.PReLU(),
            nn.Linear(256, 256),
            nn.PReLU(),
            nn.Linear(256, 2)
        )

    def forward(self, x):
        output = self.convnet(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def get_embedding(self, x):
        return self.forward(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2=None, x3=None):
        if x2 is None and x3 is None:
            return self.embedding_net(x1)
        return self.embedding_net(x1),self.embedding_net(x2),self.embedding_net(x3)

    def get_embedding(self, x):
        return self.embedding_net(x)


model_loaded = torch.load('model/tripletTAMIZH.pt')

class TomCounts_ARI:
	def __init__(self):
		pygame.mixer.init()
		pygame.init()
		pygame.font.init()
		self.det = None
		self.list = []
		self.font = pygame.font.Font("quake.TTF", 80)
		self.count = 0
		self.width = 1920
		self.height = 1080
		self.disp = pygame.display.set_mode((self.width,self.height),0,0)
		pygame.display.set_caption("MNIST Tom")
		self.img = pygame.image.load("Speak/0001.jpg")
		self.img = pygame.transform.scale(self.img,(w+120,h+160))
		self.CAPTURE_ALL()

	def blitForever(self,val=None):
		if(val!=None):
			if(val not in self.list):
				self.list = []
				self.list.append(val)
				self.playAudioARI(val)
		else:
			self.disp.blit(self.img,(560,0,0,0))
		self.disp.blit(self.imgframe,(0,0,0,0))
		text1 = self.font.render("TOM DETECTED "+str(self.det), True, white)
		text1Rect = text1.get_rect()
		text1Rect.center = (w//2,h)
		self.blittext()
		pygame.display.update()

	def blittext(self):
		text1 = self.font.render("TOM DETECTED "+str(self.det), True, white)
		text1Rect = text1.get_rect()
		text1Rect.center = (w//2+600,h+40)
		self.disp.blit(text1,text1Rect)
 
	def playAudioARI(self,op):
		i = 1
		self.det = op[9]
		pygame.mixer.music.load(op)
		pygame.mixer.music.play()
		while(True):
			img = pygame.image.load("Speak/0"+str(i).zfill(3)+".jpg")
			img = pygame.transform.scale(img,(w+120,h+160))
			self.disp.blit(img,(560,0,0,0))
			i+=1
			if(i==10):
				return
			self.disp.blit(self.imgframe,(0,0,0,0))
			self.blittext()
			pygame.display.update()

	def CAPTURE_ALL(self):
		model = torch.load('model/tripletTAMIZH.pt')
		th=0
		while True:
		    frame = cv2.resize(x[th], (w//2+50+50,h+50+80))
		    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)  # Convert RGB to BGR for OpenCV display
		    cv2.imwrite('imgframe.jpg',frame)
		    self.imgframe = pygame.image.load("imgframe.jpg")
		    image_tensor = transform(y[th])
		    image_tensor = image_tensor.unsqueeze(0)
		    emb = model_loaded(image_tensor)
		    label = index.search(emb.detach().reshape(1,-1),1)[1][0][0]
		    value = labelsd[str(labels[label])]
		    self.det = f'{labels[label]}'
		    
		    self.blitForever(f'SpeakAud/0.mp3')

		    self.blitForever()
		    for	eve in pygame.event.get():
		    	if eve.type==pygame.KEYDOWN:
		    		if eve.key==pygame.K_RIGHT:
				        if th + 1 == len(imgs):
				            th = 0
				        else:
				            th += 1

		    		elif eve.key==pygame.K_LEFT:
				        if th - 1 != -1:
				            th -= 1

		    		elif eve.key==pygame.K_ESCAPE:
		    			sys.exit()

		cv2.destroyAllWindows()


TomCounts_ARI()
