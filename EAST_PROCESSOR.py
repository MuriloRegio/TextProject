import os
import numpy as np
import tensorflow as tf
import cv2

def process(filelist, status, EAST, checkpoint_path="./Models/east_icdar2015_resnet_v1_50_rbox/"):
	with tf.get_default_graph().as_default():
		input_images = tf.placeholder(tf.float32, shape=[None, None, None, 3], name='input_images')
		global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)

		f_score, f_geometry = EAST.getModel(input_images, is_training=False)

		variable_averages = tf.train.ExponentialMovingAverage(0.997, global_step)
		saver = tf.train.Saver(variable_averages.variables_to_restore())

		with tf.Session(config=tf.ConfigProto(allow_soft_placement=True)) as sess:
			ckpt_state = tf.train.get_checkpoint_state(checkpoint_path)
			model_path = os.path.join(checkpoint_path, os.path.basename(ckpt_state.model_checkpoint_path))
			saver.restore(sess, model_path)
			i = 0
			for filename in filelist:
				im = cv2.imread(filename)[:, :, ::-1]
				im_resized, (ratio_h, ratio_w) = EAST.resize_image(im)

				timer = {'net': 0, 'restore': 0, 'nms': 0}
				score, geometry = sess.run([f_score, f_geometry], feed_dict={input_images: [im_resized]})
				boxes, timer = EAST.detect(score_map=score, geo_map=geometry, timer=timer)

				if boxes is not None:
					boxes = boxes[:, :8].reshape((-1, 4, 2))
					boxes[:, :, 0] /= ratio_w
					boxes[:, :, 1] /= ratio_h
				else:
					boxes = []

				print (score)
				# print (len(boxes))
				bbs = []
				for box in boxes:
					# to avoid submitting errors
					box = EAST.sort_poly(box.astype(np.int32))
					if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3]-box[0]) < 5:
						continue
					bbs.append(np.asarray([
						[box[0, 0], box[0, 1]], [box[1, 0], box[1, 1]], [box[2, 0], box[2, 1]], [box[3, 0], box[3, 1]],
					]))
				status["res"].append(bbs)
				i+=1
				print ("EAST finished img n# {}!".format(i))


def predict(filelist, status, imp_path='./EAST/'):
	import sys
	
	sys.path.append(os.path.realpath(imp_path))
	os.environ['CUDA_VISIBLE_DEVICES'] = "0"
	import EAST.eval as EAST
	del sys.path[-1]
	status['ready'] = 1

	try:
		process(filelist,status,EAST)
	except Exception as e:
		status["done"] = -1
		raise e
	# finally:
	# 	sys.path.remove(os.path.realpath(imp_path))
	status["done"]=1