import matplotlib.pyplot as plt
import numpy as np 
import time
from PIL import Image


#=====================================================================================
# basic sequential implemantation
#=====================================================================================
start = time.time()
rightimg = np.array(Image.open('left_bw.png'),dtype=np.int32)
leftimg = np.array(Image.open('right_bw.png'),dtype=np.int32)


rows,columns= leftimg.shape
disparityRange = 50

rad = 6
blocksize = 2*rad + 1

#initialising 3D disparity matrix
disparity = 255*np.ones(shape=(rows-2*rad,columns-2*rad,disparityRange+1))
output = np.zeros(shape=(rows-2*rad,columns-2*rad))

# padding the right image
padding = 255*np.ones(shape=(rows,disparityRange))
rightimg = np.append(rightimg,padding,axis = 1)           


# calculating the disparity matrix values 
for i in range (rad,rows-rad):
    for j in range (rad,columns-rad):
        for k in range(0,disparityRange+1):
            error =  np.abs(leftimg[i-rad:i+rad,j-rad:j+rad] - rightimg[i-rad:i+rad,j-rad+k:j+rad+k])
            disparity[i-rad,j-rad,k] = np.sum(error) + abs(k)/2
print('disparity matrix done')


# calculating the output from the disparity matrix
for i in range (0+rad,rows-rad):
    for j in range (0+rad,columns-rad):
        t = np.argmin(disparity[i-rad,j-rad,:])
        
        output[i-rad,j-rad] = np.argmin(disparity[i-rad,j-rad,:])
        if output[i-rad,j-rad] > 51:       
            output[i-rad,j-rad] = 52       

end = time.time()

print(f"elapsed time : {end-start} seconds in  basic sequential")
plt.imshow(output,cmap = "jet")
plt.colorbar()
plt.show()
print("display done")
