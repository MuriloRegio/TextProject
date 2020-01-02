import os

for line in os.popen("ps -ef | grep PNG").read().split('\n'):
	if len(line.strip()) == 0:
		continue
	line = line.split(' ')
	os.system('kill {}'.format(line[2]))
