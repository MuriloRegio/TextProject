import os
import numpy as np

def process(filelist, status, model, FOTS, with_image=False, output_img_dir=False, with_gpu=False, labels=False, output_txt_dir=False):
	i = 0
	for filename in filelist:
		polys = FOTS.predict(filename, model, with_image, output_img_dir, with_gpu, labels, output_txt_dir)
		if polys is None:
			polys = np.asarray([])
		status["res"].append(list(polys.astype(np.int32))) 
		i+=1
		print ("FOTS finished img n# {}!".format(i))
	
def predict(filelist, status, imp_path='./FOTS/', model_path="Models/retrained_model.pth.tar"):
	import sys
	import pathlib

	sys.path.append(os.path.realpath(imp_path))
	os.environ['CUDA_VISIBLE_DEVICES'] = ""
	import FOTS.eval as FOTS
	model = FOTS.load_model(model_path,False)
	del sys.path[-1]
	status['ready'] = 1
	
	try:
		process(map(pathlib.Path,filelist), status, model, FOTS)
	except Exception as e:
		status["done"] = -1
		raise e
	# finally:
		# sys.path.remove(os.path.realpath(imp_path))
	status["done"] = 1