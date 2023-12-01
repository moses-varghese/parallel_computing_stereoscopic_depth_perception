import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np 
import time
from PIL import Image

# ==================================================================
# improved sequntial 
# ==================================================================

start = time.time()
rightimg = np.array(Image.open('left_bw.png'),dtype=np.int32)
leftimg = np.array(Image.open('right_bw.png'),dtype=np.int32)


rows,columns= leftimg.shape
disparityRange = 50

rad = 2
blocksize = 2*rad + 1
# ==================================================================
#initialising 3D disparity matrix
# ==================================================================

disparity = 255*np.ones(shape=(rows-2*rad,columns-2*rad,disparityRange+1))
output = np.zeros(shape=(rows-2*rad,columns-2*rad))
# ==================================================================
# padding the right image
# ==================================================================
padding = 255*np.ones(shape=(rows,disparityRange))
rightimg = np.append(rightimg,padding,axis = 1)           
# ==================================================================
# Initialising Gradients and orientation
# ==================================================================
gradientleft_x = 255.0*np.ones(shape=(rows,columns),dtype=np.float32)
gradientleft_y = 255.0*np.ones(shape=(rows,columns),dtype=np.float32)
gradientright_x = 255.0*np.ones(shape=(rows,columns+disparityRange),dtype=np.float32)
gradientright_y = 255.0*np.ones(shape=(rows,columns+disparityRange),dtype=np.float32)
Oleft  = 255.0*np.ones(shape=(rows,columns),dtype=np.float32)
Oright = 255.0*np.ones(shape=(rows,columns+disparityRange))


# ==================================================================
# create gradient matrixes and orientation matrix
# ==================================================================

for i in range (rows):
    for j in range(columns):
        if  j ==0 :
            gradientleft_x[i,j] = leftimg[i,j+1]-leftimg[i,j]
            gradientright_x[i,j] = rightimg[i,j+1] - rightimg[i,j]
        elif  j ==columns-1 :
            resl = leftimg[i,j] - leftimg[i,j-1]
            gradientleft_x[i,j] = resl
            gradientright_x[i,j] = rightimg[i,j] - rightimg[i,j-1]
        else :
            resl = (leftimg[i,j+1] - leftimg[i,j-1])/2.0    
            gradientleft_x[i,j] = resl
            gradientright_x[i,j] = (rightimg[i,j+1] - rightimg[i,j-1])/2.0


        if i == 0 :
            gradientleft_y[i,j] = leftimg[i+1,j] - leftimg[i,j]
            gradientright_y[i,j] = rightimg[i+1,j] - rightimg[i,j]

        elif i == rows-1 :
            gradientleft_y[i,j] = leftimg[i,j] - leftimg[i-1,j]
            gradientright_y[i,j] = rightimg[i,j] - rightimg[i-1,j]

        else :
            gradientleft_y[i,j] = (leftimg[i+1,j] - leftimg[i-1,j])/2.0
            gradientright_y[i,j] = (rightimg[i+1,j] - rightimg[i-1,j])/2.0

        if gradientleft_x[i,j] < 00.1 and gradientleft_x[i,j] >-0.001:
            Oleft[i,j]  = 0.0
        else:
            Oleft[i,j] = gradientleft_y[i,j]/gradientleft_x[i,j]

        if gradientright_x[i,j] < 00.1 and gradientright_x[i,j] >-0.001 :
            Oright[i,j]  = 0.0
        else:
            Oright[i,j] = gradientright_y[i,j]/gradientright_x[i,j]


# ==================================================================
# calculating the disparity matrix values 
# ==================================================================

for i in range (rad,rows-rad):
    for j in range (rad,columns-rad):
        for k in range(0,disparityRange+1):            
            error =  np.abs(leftimg[i-rad:i+rad,j-rad:j+rad] - rightimg[i-rad:i+rad,j-rad+k:j+rad+k])
            error += np.abs(gradientleft_x[i-rad:i+rad,j-rad:j+rad] - gradientright_x[i-rad:i+rad,j-rad+k:j+rad+k])
            error += np.abs(gradientleft_y[i-rad:i+rad,j-rad:j+rad] - gradientright_y[i-rad:i+rad,j-rad+k:j+rad+k])
            error += np.abs(Oleft[i-rad:i+rad,j-rad:j+rad] - Oright[i-rad:i+rad,j-rad+k:j+rad+k])
            disparity[i-rad,j-rad,k] = np.sum(error) + abs(k)/2 
print('disparity matrix done')

# ==================================================================
# calculating the output from the disparity matrix
# ==================================================================

for i in range (0+rad,rows-rad):
    for j in range (0+rad,columns-rad):
        t = np.argmin(disparity[i-rad,j-rad,:])
        
        output[i-rad,j-rad] = np.argmin(disparity[i-rad,j-rad,:])
        if output[i-rad,j-rad] > 51:       
            output[i-rad,j-rad] = 52       
end = time.time()


print(f"elapsed time : {end-start} seconds in improved sequential")
plt.imshow(output,cmap = "jet")
plt.colorbar()
plt.show()
print("display done")
