import cv2
import numpy as np
from PIL import Image
from filtering import nms

class text_candidates:
	def __init__(self, binarize = lambda x : cv2.cvtColor(x, cv2.COLOR_RGB2GRAY), cfilter = nms):
	# def __init__(self, binarize = lambda x : cv2.Canny(x,100,300)):
		self.mser = cv2.MSER_create()
		self.binarize = binarize
		self.cfilter = cfilter

	def filter_candidates(self, regions, data):
		tmp = []

		regions = np.asarray(regions)
		data = np.asarray(data)

		for i,BB in enumerate(data):
			try:
				x, y, w, h = BB
			except:
				print (BB)
				1/0

			tmp.append([x, y, x + w, y + h])

		tmp = np.asarray(tmp)
		picks = self.cfilter(tmp)

		tmp,tmp2 = [],[]
		for r,d in zip(regions[picks],data[picks]):
			x, y, w, h = d

			if w < 9 or h < 9:
				continue

			tmp.append(d)
			tmp2.append(r)


		return np.asarray(tmp2),np.asarray(tmp)

	def getCandidates(self,img,filter_ = False):
		if type(img) is str:
			img = cv2.imread(img)

		if len(img.shape) == 3:
			gray = self.binarize(img)
		else:
			gray = img

		regions, BB = self.mser.detectRegions(gray)

		if filter_:
			regions, BB = self.filter_candidates(regions,BB)

		return regions, BB

	def yieldCandidates(self,img,filter_ = True):
		regions, BB = self.getCandidates(img,filter_)
		for (x,y,w,h) in BB:
			yield img[y:y+h, x:x+w]

	def getMasked(self,img,data = None, filter_ = False):
		if data is None:
			regions, BB = self.getCandidates(img,filter_)
			data = zip(regions,BB)

		res = []
		for r,(x,y,w,h) in data:
			r = np.asarray(map(lambda var : (var[0]-x,var[1]-y), r))
			hull = cv2.convexHull(r.reshape(-1,2))
			sub_im = img[y:y+h, x:x+w]
	
			mask = np.zeros(sub_im.shape[:2], dtype=np.uint8)
			cv2.drawContours(mask, [hull], -1, (255, 255, 255), -1)
			text_only = cv2.bitwise_and(sub_im, sub_im, mask=mask)
			res.append(text_only)
		return res


	def visualize(self,img,regions):
		if type(img) is str:
			img = cv2.imread(img)
		vis = img.copy()

		hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

		cv2.polylines(vis, hulls, 1, (0, 255, 0))

		Image.fromarray(vis).show()

	def visualize_text_only(self,img,regions):
		if type(img) is str:
			img = cv2.imread(img)
		vis = img.copy()

		hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions]

		cv2.polylines(vis, hulls, 1, (0, 255, 0))

		mask = np.zeros((img.shape[0], img.shape[1], 1), dtype=np.uint8)

		for contour in hulls:
			cv2.drawContours(mask, [contour], -1, (255, 255, 255), -1)

		#this is used to find only text regions, remaining are ignored
		text_only = cv2.bitwise_and(img, img, mask=mask)

		# cv2.imshow("text only", text_only)
		Image.fromarray(text_only).show()