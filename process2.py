

import time
import pickle

for i in range(3):
	time.sleep(2)
	fobj = open('_out'+str(i)+'.pkl', 'rb')
	print(pickle.load(fobj))
	fobj.close()

	
	