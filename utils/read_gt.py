import numpy as np

def implies(p,q):
	return (p and q) or (not p)

class HiddenPrints:
	def __enter__(self):
		import sys
		import os
		self._original_stdout = sys.stdout
		sys.stdout = open(os.devnull, 'w')

	def __exit__(self, exc_type, exc_val, exc_tb):
		import sys
		sys.stdout.close()
		sys.stdout = self._original_stdout

class reader:
	def __init__(self, path = None, extension = '.txt', size = 8, length = 1000):
		self.path = path
		self.extension = extension
		self.size = size
		self.i = 0
		self.length = length

	def __iter__(self):
		return self

	def next(self):
		if self.i < self.length:
			i = self.i
			self.i += 1
			return getICDAR15(i+1)
		else:
			self.i = 0
			raise StopIteration()

	def readfile(self, iden):
		assert implies(self.path is None, type(iden) is str)
		assert implies(self.path is not None, type(iden) is int)

		if type(iden) is str:
			file = iden
		else:
			file = self.path + str(iden) + self.extension

		with open(file, 'r') as infile:
			obj = {}
			i = 0
			for line in infile:
				line = line.replace("\xef\xbb\xbf","")
				a = line.split(',')
				formatedKey = ','.join(a[self.size:]) [:-2] + " {}".format(str(i))
				i+=1
				try:
					obj[formatedKey] = map(int,a[:self.size])
				except:
					print line
					1/0
		return obj

def getICDAR15(i, img_dir, gt_dir, region=False):
	r = reader()
	def openimg(i_):
		from PIL import Image
		import os

		img = np.asarray(Image.open(os.path.join(img_dir,'img_{}.jpg'.format(i_))))
		gt = os.path.join(gt_dir,'gt_img_{}.txt'.format(i_))

		return img,gt

	img, gt_path = openimg(i)
	objs = r.readfile(gt_path)

	if region:
		reg = {}
		for k,v in objs.items():
			v = np.asarray([[v[i],v[i+1]] for i in range(0,len(v),2)])
			reg[k]=v
		return reg,img

	a = []
	for _,v in objs.items():
		xs = [v[i] for i in range(0,len(v),2)]
		ys = [v[i] for i in range(1,len(v),2)]
		p1 = min(xs),min(ys)
		p2 = max(xs),max(ys)
		a.append(p1+p2)

	return a,img

class COCO_manager:
	def __init__(self):
		with HiddenPrints():
			import coco.coco_text as ct
			self.ct = ct.COCO_Text('/home/davint-1/datasets/COCO_Text.json')
			self.imgIds = lambda : self.ct.getImgIds(imgIds=self.ct.train, catIds=[('legibility','legible')])
			self.size = len(self.imgIds())
			self.i = 0

	def get(self,i=None, ID=None):
		if i is None and ID is None:
			raise ValueError("Inform either an index or ID")

		formatter = lambda (x,y,w,h) : map(int,[x,y,x+w,y+h])
		from os.path import join
		from PIL import Image
		with HiddenPrints():
			if ID is None:
				img = self.ct.loadImgs(self.imgIds()[i])[0]
			else:
				img = self.ct.loadImgs(ID)[0]

			im = np.asarray(Image.open(join('/home/davint-1/datasets/train2014',img['file_name'])))
			annIds = self.ct.getAnnIds(imgIds=img['id'])
			anns = self.ct.loadAnns(annIds)
			bbox = [formatter(x['bbox']) for x in anns]
		return bbox, im

	def __contains__(self,ID):
		return ID in self.imgIds()

	def __iter__(self):
		return self

	def next(self):
		if self.i < self.size:
			i = self.i
			self.i += 1
			return self.get(i)
		else:
			self.i = 0
			raise StopIteration()
	
if __name__ == "__main__":
	i = 500
	c = COCO_manager()
	a,img = c.get(i)
	# print c.size
	# a,img = get_icdar15(i)
	# print a

	from candidates import drawBB
	drawBB(img,a,True)