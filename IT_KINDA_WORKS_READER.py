
from PIL import Image
import numpy as np
import pytesseract
import cv2

whitelist = ''.join([chr(ord(x)+i) for x in "aA" for i in range(ord('z')-ord('a')+1)])
ratio = 10

def getText(im):
	return pytesseract.image_to_string(im, config='--oem 1 --psm 7 -c tessedit_char_whitelist={}'.format(whitelist))

def individually(im):
	from utils.candidates import text_candidates
	t = text_candidates()

	for sub_im in t.yieldCandidates(im):
		print (getText(sub_im))

def process(im):
	im = cv2.resize(im,(0,0),fx=ratio,fy=ratio)
	im = cv2.dilate(im, None)
	im = cv2.dilate(im, None)
	im = cv2.dilate(im, None)
	im = cv2.cvtColor(im,cv2.COLOR_RGB2GRAY)
	individually(im)
	_,im = cv2.threshold(im,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	im = cv2.erode(im, None)
	im = cv2.erode(im, None)
	# im = cv2.erode(im, None)
	im = cv2.bitwise_not(im)
	im = cv2.resize(im,(0,0),fx=1./ratio,fy=1./ratio)
	return im

def to_text(img, coord_list):
	template = lambda l,axis : (max(min(l)-5,0), min(max(l)+5, img.shape[axis]))
	# template = lambda l : (min(l), max(l))
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

	for poly in coord_list:
		xs, ys = zip(*poly)

		# xmin,xmax = template(xs)
		# ymin,ymax = template(ys)
		xmin,xmax = template(xs,1)
		ymin,ymax = template(ys,0)

		w = ymax-ymin
		h = xmax-xmin
		center = w/2,h/2

		sub_im = cv2.fillConvexPoly(np.zeros((w,h),dtype=np.uint8), poly-(xmin,ymin), 255)

		sub_im = cv2.bitwise_and(img[ymin:ymax,xmin:xmax],img[ymin:ymax,xmin:xmax],mask=sub_im)

		angle = cv2.minAreaRect(poly)[-1]
		print (angle)

		if angle < -45:
			angle =  (90 + angle)
			print (angle)
			
		M = cv2.getRotationMatrix2D(center, angle, 1.0)
		sub_im = cv2.warpAffine(sub_im, M, (h, w), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

		for i in range(sub_im.shape[0]):
			if (sub_im[i]>0).any():
				r = i
				break

		sub_im = sub_im[r:]

		sub_im = process(sub_im)

		sub_im = Image.fromarray(sub_im)
		# sub_im.show()
		
		text = getText(sub_im)
		print ('-->',text)
		if len(text)==0:
			sub_im.show()
			Image.fromarray(img[ymin:ymax,xmin:xmax]).show()
		# exit(0)

if __name__ == "__main__":
	import os
	ims_path = "~/datasets/ICDAR15_TEST"

	import sys

	args = sys.argv[1:]
	start = 1
	if len(args) > 0:
		start = int(args[0])

	with open("dump.dat",'r') as log:
		from numpy import array, int32
		results = eval(log.read())
	
	format_home = lambda x : x.replace('~', os.popen('echo ~').read()[:-1])
	ims_path = format_home(ims_path)
	filelist = list(map(lambda x : os.path.realpath(os.path.join(ims_path,x)),os.listdir(ims_path)))

	for i in range(start,len(results[0])):
		print ('-----------------------------------------')
		print ('Imagem',i+1)
		img = np.asarray(Image.open(filelist[i]))
		to_text(img,results[0][i])
		print ('-----------------------------------------')
		break