from numba import cuda 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time 


@cuda.jit
def kerneldisparity (d_leftimg, d_rightimg , d_disparity):
    '''
    Input:
    d_leftimg: left image matrix stored in the device; 
    d_righttimg: right image matrix stored in the device
    Output:
    d_disparity: Disparity matrix 
    '''
    rows,columns = d_leftimg.shape
    blocksize = 2*rad + 1
    # bounds check
    i,j = cuda.grid(2)
    if (i >= rows-rad) or (j >= columns-rad):
        return
    minr = max(0,i-rad)
    maxr = min(rows-1,i+rad)
    minc = max(0,j-rad)
    maxc = min(columns-1,j+rad)
    for k in range(disparityRange+1):            
        error = 0 
        for ii in range(minr,maxr):
            for jj in range(minc, maxc):
                error += abs(d_leftimg[ii,jj] - d_rightimg[ii,jj+k+disparityRange])
        d_disparity[i-rad,j-rad,k] = error + abs(k)/2 
    
# ======================================================================
def pardisparity():
    '''
    Wrapper function to compute the disparity matrix
    '''
    start = time.time()
    rightimg = np.array(Image.open('left_bw.png'))
    leftimg  = np.array(Image.open('right_bw.png'))
    
    rows,columns= leftimg.shape

    #initialising 3D disparity matrix
    disparity = 255*np.ones(shape=(rows-2*rad,columns-2*rad,disparityRange+1))
    output = np.zeros(shape=(rows-2*rad,columns-2*rad))

    # padding the right image
    padding = 255*np.ones(shape=(rows,disparityRange))
    rightimg = np.append(padding,rightimg,axis = 1)           
    rightimg = np.append(rightimg,padding,axis = 1)
    
    d_leftimg  = cuda.to_device(leftimg)
    d_rightimg = cuda.to_device(rightimg)
    d_disparity  =  cuda.to_device(disparity)

    TPBx = 8
    TPBy = 8
    BPGx = (rows +TPBx-1)//TPBx
    BPGy = (columns +TPBy-1)//TPBy
    kerneldisparity[[BPGx,BPGy], [TPBx, TPBy]](d_leftimg, d_rightimg , d_disparity)
    disparity = d_disparity.copy_to_host()

    # calculating the output from the disparity matrix
    for i in range (0+rad,rows-rad):
        for j in range (0+rad,columns-rad):
            t = np.argmin(disparity[i-rad,j-rad,:])
        
            output[i-rad,j-rad] = np.argmin(disparity[i-rad,j-rad,:])
            if output[i-rad,j-rad] > 51:       
                output[i-rad,j-rad] = 52       

    end = time.time()
    print(f"elapsed time : {end-start} seconds in parallel")

    return output


disparityRange = 50
rad = 6
output = pardisparity()
output = pardisparity()
output = pardisparity()


plt.imshow(output,cmap = "jet")
plt.colorbar()
plt.show()
