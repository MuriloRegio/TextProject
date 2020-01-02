import sys
import os

import argparse


parser = argparse.ArgumentParser(description="Combine methods' results.")
parser.parse_args()

import execution_parameters as args

ims_path = args.ims_path
gts_path = args.gts_path
thresh 	 = args.thresh
ext 	 = args.ext
outname  = args.outname
logfile  = args.logfile
dumpfile = args.dumpfile
implementations_list = args.implementations_list

log = open(logfile+'.dat','w')
stdout = sys.stdout
sys.stdout = log


format_home = lambda x : x.replace('~', os.popen('echo ~').read()[:-1])
ims_path = format_home(ims_path)
gts_path = format_home(gts_path)


files = []
gts = []
for filename in os.listdir(ims_path):
	name = filename[:filename.index('.')]

	files.append(os.path.realpath(os.path.join(ims_path,name+".jpg")))
	gts.append(os.path.realpath(os.path.join(gts_path,"gt_"+name+".txt")))


def predict(filelist):
	import time
	import _thread
	
	status_list = []

	for predictor in predictors:
		status_list.append({'res':[], "done":0, "ready":0})

		_thread.start_new_thread(
			predictor,
			(filelist,status_list[-1])
		)
		while not status_list[-1]["ready"]:
			time.sleep(1)


	while 1:
		end = 1

		for status in status_list:
			if status["done"] == -1:
				exit(0)
			end = end and status["done"]

		if end:
			break
		time.sleep(60)

	return [status["res"] for status in status_list]


predictors = []

for implementation in implementations_list:
	code = """if 1:
	import {0}
	predictors.append({0}.predict)
	""".format(implementation)

	obj = compile(code,'','exec')
	exec(obj)





results = predict(files)

if dumpfile:
	with open(dumpfile+".dat",'w') as log:
		log.write(str(results))
	# from numpy import array, int32
	# results = eval(log.read())



import joiner
results = joiner.process(results)




res = {"TP":0, "FP":0, "FN":0}
from FOTS.utils.bbox import Toolbox
from FOTS.eval import load_annotation
import pathlib


or i,abs_gt_path in enumerate(gts):
	# boxes = list(results[0][i])+list(results[1][i])
	# boxes = sum([list(x[i]) for x in results], [])
	boxes = results[i]
	annot = load_annotation(pathlib.Path(abs_gt_path))

	true_pos, false_pos, false_neg = Toolbox.comp_gt_and_output(boxes, annot, thresh)
	res["TP"] += true_pos
	res["FP"] += false_pos
	res["FN"] += false_neg

	i+=1
	print ("Finished annot n# {}!".format(i))

sys.stdout = stdout

with open(outname+".txt",'w') as outfile:
	outfile.write('Brute Values:\n')
	for k,v in res.items():
		outfile.write('\t{}: {}\n'.format(k,v))
	outfile.write('\n\n')

	outfile.write('Processed Values:\n')

	P = res["TP"]/(res["TP"]+res["FP"])
	outfile.write('\tP: {}\n'.format(P))

	R = res["TP"]/(res["TP"]+res["FN"])
	outfile.write('\tR: {}\n'.format(R))

	outfile.write('\tF: {}\n'.format(2*P*R/(P+R)))


# os.system("shutdown now")
