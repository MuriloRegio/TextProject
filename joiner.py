from PIL import Image
import numpy as np
import cv2

thresh = .5

def intersection(p1,p2):
	def template(x,axis):
		x1s,y1s=zip(*p1)
		x2s,y2s=zip(*p2)
		return int(eval("{0}({0}({1}1s), {0}({1}2s))".format(x,axis)))


	draw = lambda x : cv2.fillConvexPoly(np.zeros((xmax-xmin,ymax-ymin), dtype=np.uint8), x-(xmin,ymin), (1))

	xmin = template("min",'x')
	xmax = template("max",'x')
	ymin = template("min",'y')
	ymax = template("max",'y')

	draw1 = draw(p1)
	a1 = draw1.sum()

	draw2 = draw(p2)
	a2 = draw2.sum()

	intersec = 1.*cv2.bitwise_and(draw1,draw2).sum()
	union = cv2.bitwise_or(draw1,draw2).sum()

	if intersec/union<thresh:
		return

	# print 'Passed with IoU',intersec/union

	l1 = list(p1)
	l2 = list(p2)

	new = []

	for i,(x1,y1) in enumerate(l1):
		closest = float('inf'),None
		for j,(x2,y2) in enumerate(l2):
			distance = (x1-x2)**2 + (y1-y2)**2

			if distance<closest[0]:
				closest=distance,j

		x,y = l2[closest[1]]
		del l2[closest[1]]
		new.append([round((v1+v2)/2) for v1,v2 in [[x,x1],[y,y1]]])
		# print (x,y),(x1,y1),l1[i]

	return np.asarray(new, dtype=np.int32)

def process(results):
	assert all([len(results[i])==len(results[(i+1)%len(results)]) for i in range(len(results))])

	changed = []
	final = []
	for n in range(len(results[0])):
		boxes = [list(x[n]) for x in results]
		tmp = []
		cur = []

		for i in range(len(boxes)):
			for i1 in range(len(boxes[i])-1,-1,-1):
				c = 1
				p1 = boxes[i][i1]

				for j in range(len(boxes)):
					if i == j:
						continue
					for j1 in range(len(boxes[j])-1,-1,-1):
						p2 = boxes[j][j1]

						new = intersection(p1,p2)
						if new is None:
							continue

						del boxes[j][j1]
						p1 = new
						c+=1

				del boxes[i][i1]
				tmp.append(p1)
				cur.append(c)
		changed.append(cur)
		print (cur)
		final.append(tmp)
	return final

if __name__ == "__main__":
	with open("log.dat",'r') as log:
		from numpy import array, int32
		results = eval(log.read())

	print ([[len(x) for x in y] for y in results])
	final = process(results)
	print ([len(x) for x in final])

	import os
	path = '/home/davint-1/datasets/ICDAR15_TEST/'
	files = os.listdir(path)
	for i in range(5):
		im = np.asarray(Image.open(os.path.join(path,files[i])))

		mask = np.zeros(im.shape[:2],dtype=np.uint8)
		from copy import copy
		sub_im = copy(im)
		for pts in final[i]:
			# mask = cv2.fillConvexPoly(mask, pts, 255)
			# print pts.shape
			sub_im = cv2.polylines(sub_im, [pts], 1, (0,255,0))
		# sub_im = cv2.bitwise_and(im,im,mask=mask)

		for polys in [x[i] for x in results]:
			im = cv2.polylines(im, polys, 1, (0,255,0))

		Image.fromarray(im).show()
		Image.fromarray(sub_im).show()
		break