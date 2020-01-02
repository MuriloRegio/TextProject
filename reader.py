
from PIL import Image
import numpy as np
import pytesseract
import cv2

def getMask(im):
	if len(im.shape) == 3:
		tmp = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
	else:
		from copy import copy
		tmp = copy(im)

	s = tmp.sum()
	o = -1
	its = 0
	while s != o:
		o = s
		_,mask = cv2.threshold(tmp,0,255,cv2.THRESH_OTSU)
		tmp = cv2.bitwise_and(tmp,mask)
		s = tmp.sum()
		its +=1
	print its
	# return cv2.bitwise_and(tmp,tmp,mask=mask)
	# return cv2.bitwise_and(im,im,mask=mask)
	_,mask = cv2.threshold(tmp,0,255,cv2.THRESH_BINARY)
	return mask

def resize(im):
	print im.shape
	from math import ceil
	ratio_x = ceil(200./im.shape[1])
	ratio_y = ceil(100./im.shape[0])
	return cv2.resize(im,(0,0),fx=ratio_x,fy=ratio_y)


def process(im):
	# sub_im = cv2.resize(sub_im,(0,0),fx=5,fy=5)
	# sub_im = np.asarray(Image.fromarray(sub_im).resize((300,100),Image.BICUBIC))
	im = resize(im)
	# sub_im = cv2.dilate(sub_im, None)
	im = getMask(im)
	im = cv2.erode(im, None)
	# sub_im = cv2.erode(sub_im, None)
	im = cv2.bitwise_not(im)
	return im

def to_text(img, coord_list):
	template = lambda l,axis : (max(min(l)-5,0), min(max(l)+5, img.shape[axis]))
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	for n,poly in enumerate(coord_list):
		print ''
		xs, ys = zip(*poly)

		xmin,xmax = template(xs,1)
		ymin,ymax = template(ys,0)

		w = ymax-ymin
		h = xmax-xmin
		center = w/2,h/2

		sub_im = cv2.fillConvexPoly(np.zeros((w,h),dtype=np.uint8), poly-(xmin,ymin), 255)
		sub_im = cv2.bitwise_and(gray[ymin:ymax,xmin:xmax],gray[ymin:ymax,xmin:xmax],mask=sub_im)

		angle = cv2.minAreaRect(poly)[-1]

		M = cv2.getRotationMatrix2D(center, angle, 1.0)
		print angle
		# Image.fromarray(sub_im).show()
		# sub_im = cv2.warpAffine(sub_im, M, (h, w), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
		sub_im = cv2.warpAffine(sub_im, M, (h, w), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_REPLICATE)
		# Image.fromarray(sub_im).show()

		r = 0
		for i in range(sub_im.shape[0]):
			if (sub_im[i]>0).any():
				r = i
				break

		sub_im = sub_im[r:]

		tmp = sub_im.shape
		sub_im = Image.fromarray(sub_im)
		
		text = pytesseract.image_to_string(sub_im, config='--psm 7 -c tessedit_char_blacklist=(')
		print n+1,text, tmp
		if len(text)==0:
			sub_im.show()
			Image.fromarray(img[ymin:ymax,xmin:xmax]).resize((100,100), Image.BICUBIC).show()
		raw_input('')
	print ''
		# exit(0)

if __name__ == "__main__":
	import os
	ims_path = "~/datasets/ICDAR15_TEST"

	with open("dump.dat",'r') as log:
		from numpy import array, int32
		results = eval(log.read())
	
	format_home = lambda x : x.replace('~', os.popen('echo ~').read()[:-1])
	ims_path = format_home(ims_path)
	filelist = map(lambda x : os.path.realpath(os.path.join(ims_path,x)),os.listdir(ims_path))

	for i in range(100,len(results[0])):
		print '-----------------------------------------'
		print 'Imagem',i+1
		img = np.asarray(Image.open(filelist[i]))
		to_text(img,results[0][i])
		print '-----------------------------------------'
		break