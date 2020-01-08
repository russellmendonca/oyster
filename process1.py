

import time
import pickle

_list = []
for i in range(3):
	time.sleep(1)
	_list.append(i)
	print('process1 running')
	fobj = open('_out'+str(i)+'.pkl', 'wb')
	pickle.dump(_list, fobj)
	fobj.close()
	
	