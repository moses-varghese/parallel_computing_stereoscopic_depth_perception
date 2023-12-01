from numba import cuda 
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import time 


# ======================================================================
# x gradient calculator kernel
# ======================================================================
@cuda.jit
def calc_gradientx(d_gradientoutput, d_img, rows,columns):
    '''
    Inputs:
    d_img: image matrix stored on the GPU
    rows, columns: rows and columns in the d_img matrix

    Output:
    d_gradientoutput: Computed Gradient of pixel intensity in the x direction
    
    '''
    i,j = cuda.grid(2)
    # check if i or j are out of critetria
    if (i>= rows) or (j >=columns):
        return 
    # set the radtion for computing the gradient
    RAD = RAD1
    NX, NY = cuda.blockDim.x, cuda.blockDim.y

    # define indices for threads and shared array
    t_i, t_j = cuda.threadIdx.x, cuda.threadIdx.y
    sh_i, sh_j = t_i + RAD, t_j + RAD
    sh_img = cuda.shared.array(shape = (SH_N,SH_N), dtype = np.int32)

    #Load regular values
    sh_img[sh_i, sh_j] = d_img[i, j]


    #tests for bounds checking before loads from global array
    left = i - RAD > 0
    right = i + NX < rows
    up = j - RAD > 0
    down = j + NY < columns


    #Halo edge values
    if t_i < RAD:
        if left:
            sh_img[sh_i - RAD, sh_j] = d_img[i - RAD, j]
        if right:
            sh_img[sh_i + NX , sh_j] = d_img[i + NX , j]

    if t_j < RAD:
        if up:
            sh_img[sh_i, sh_j - RAD] = d_img[i, j - RAD]
        if down:
            sh_img[sh_i, sh_j + NY ] = d_img[i, j + NY ]



    #Halo corner values
    if t_i < RAD and t_j < RAD:
        #upper left
        if left and up:
            sh_img[sh_i - RAD, sh_j - RAD] = d_img[i - RAD, j - RAD]
        #upper right
        if right and up:
            sh_img[sh_i + NX, sh_j - RAD] = d_img[i + NX, j - RAD]
        #lower left
        if left and down:
            sh_img[sh_i - RAD, sh_j + NY] = d_img[i - RAD, j + NY]
        #lower right
        if right and down:
            sh_img[sh_i + NX, sh_j + NX] = d_img[i + NX, j + NY]


    #make sure shared array is fully loaded before read
    cuda.syncthreads()


    if j ==0 :
        d_gradientoutput[i,0] =  sh_img[sh_i,sh_j+1] - sh_img[sh_i,sh_j]
    elif  j == (columns-1) :
        d_gradientoutput[i,j] = sh_img[sh_i,sh_j] - sh_img[sh_i,sh_j-1]
    else :
        d_gradientoutput[i,j] = (sh_img[sh_i,sh_j+1] - sh_img[sh_i,sh_j-1])/2.0



# ======================================================================
# y gradient calculator kernel
# ======================================================================
@cuda.jit
def calc_gradienty(d_gradientoutput, d_img, rows,columns):
    '''
    Inputs:
    d_img: image matrix stored on the GPU
    rows, columns: rows and columns in the d_img matrix

    Output:
    d_gradientoutput: Computed Gradient of pixel intensity in the y direction
    
    '''
    i,j = cuda.grid(2)
    # check if i or j are out of critetria
    if (i>= rows) or (j >=columns):
        return 
    # set the radtion for computing the gradient
    RAD = RAD1
    NX, NY = cuda.blockDim.x, cuda.blockDim.y

    # define indices for threads and shared array
    t_i, t_j = cuda.threadIdx.x, cuda.threadIdx.y
    sh_i, sh_j = t_i + RAD, t_j + RAD
    sh_img = cuda.shared.array(shape = (SH_N,SH_N), dtype = np.int32)

    #Load regular values
    sh_img[sh_i, sh_j] = d_img[i, j]


    #tests for bounds checking before loads from global array
    left = i - RAD > 0
    right = i + NX < rows
    up = j - RAD > 0
    down = j + NY < columns


    #Halo edge values
    if t_i < RAD:
        if left:
            sh_img[sh_i - RAD, sh_j] = d_img[i - RAD, j]
        if right:
            sh_img[sh_i + NX , sh_j] = d_img[i + NX , j]

    if t_j < RAD:
        if up:
            sh_img[sh_i, sh_j - RAD] = d_img[i, j - RAD]
        if down:
            sh_img[sh_i, sh_j + NY ] = d_img[i, j + NY ]



    #Halo corner values
    if t_i < RAD and t_j < RAD:
        #upper left
        if left and up:
            sh_img[sh_i - RAD, sh_j - RAD] = d_img[i - RAD, j - RAD]
        #upper right
        if right and up:
            sh_img[sh_i + NX, sh_j - RAD] = d_img[i + NX, j - RAD]
        #lower left
        if left and down:
            sh_img[sh_i - RAD, sh_j + NY] = d_img[i - RAD, j + NY]
        #lower right
        if right and down:
            sh_img[sh_i + NX, sh_j + NX] = d_img[i + NX, j + NY]


    #make sure shared array is fully loaded before read
    cuda.syncthreads()


    if i ==0 :
        d_gradientoutput[i,0] =   sh_img[sh_i+1,sh_j] - sh_img[sh_i,sh_j]
    elif  i == (rows-1) :
        d_gradientoutput[i,j] =   sh_img [sh_i,sh_j] - sh_img[sh_i-1,sh_j]
    else :
        d_gradientoutput[i,j] = ( sh_img[sh_i+1,sh_j] - sh_img[sh_i-1,sh_j])/2.0
    
# ======================================================================
# orientation kernel
# ======================================================================
@cuda.jit
def calc_orientation(d_gradx, d_grady,d_o, rows,columns):
    '''
    Inputs: 
    d_gradx, d_grady: gradient matrices in the x and y directions
    rows, columns: rows and columns in the d_img matrix

    Output:
    d_o: orientation matrix    
    '''
    i,j = cuda.grid(2)
    if (i >= rows) or (j >= columns):
        return
    if d_gradx[i,j] < 0.001 and d_gradx[i,j] > -0.001:
        d_o[i,j] = 0
    else :
        d_o[i,j] = d_grady[i,j]/(d_gradx[i,j]+0.01)







# ======================================================================
# disparity kernel
# ======================================================================
@cuda.jit
def kerneldisparity (d_leftimg, d_rightimg , d_disparity,d_gradientleft_x, d_gradientleft_y, d_gradientright_x, d_gradientright_y ,d_Oleft, d_Oright):
    '''
    Inputs:
    d_leftimg, d_rightimg: image matrices on the GPU, left and right
    d_gradientleft_x, d_gradientleft_y: Gradient matrices on the GPU, left image, x and y directions
    d_gradientright_x, d_gradientright_y:  Gradient matrices on the GPU, right image, x and y directions
    d_Oleft, d_Oright: Orientation matrices, left and right images

    Output:
    d_disparity: disparity matrix
    '''
    rows,columns = d_leftimg.shape
    rad = RAD2
    blocksize = 2*rad + 1
    # bounds check
    i,j = cuda.grid(2)
    if (i >= rows-rad) or (j >= columns-rad):
        return

    # set the radtion for computing the gradient
    RAD = RAD2
    NX, NY = cuda.blockDim.x, cuda.blockDim.y

    # define indices for threads and shared array
    t_i, t_j = cuda.threadIdx.x, cuda.threadIdx.y
    sh_i, sh_j = t_i + RAD, t_j + RAD
    sh_leftimg = cuda.shared.array(shape = (SH_N2,SH_N2), dtype = np.int32)
    #Load regular values
    sh_leftimg[sh_i, sh_j] = d_leftimg[i, j]

    
    #tests for bounds checking before loads from global array
    left = i - RAD > 0
    right = i + NX < rows
    up = j - RAD > 0
    down = j + NY < columns
    #Halo edge values
    if t_i < RAD:
        if left:
            sh_leftimg[sh_i - RAD, sh_j] = d_leftimg[i - RAD, j]
        if right:
            sh_leftimg[sh_i + NX , sh_j] = d_leftimg[i + NX , j]
    if t_j < RAD:
        if up:
            sh_leftimg[sh_i, sh_j - RAD] = d_leftimg[i, j - RAD]
        if down:
            sh_leftimg[sh_i, sh_j + NY ] = d_leftimg[i, j + NY ]
    #Halo corner values
    if t_i < RAD and t_j < RAD:
        #upper left
        if left and up:
            sh_leftimg[sh_i - RAD, sh_j - RAD] = d_leftimg[i - RAD, j - RAD]

            
        #upper right
        if right and up:
            sh_leftimg[sh_i + NX, sh_j - RAD] = d_leftimg[i + NX, j - RAD]

        #lower left
        if left and down:
            sh_leftimg[sh_i - RAD, sh_j + NY] = d_leftimg[i - RAD, j + NY]

        #lower right
        if right and down:
            sh_leftimg[sh_i + NX, sh_j + NX] = d_leftimg[i + NX, j + NY]



    #make sure shared array is fully loaded before read
    cuda.syncthreads()


    minr = max(0,i-rad)
    maxr = min(rows-1,i+rad)
    minc = max(0,j-rad)
    maxc = min(columns-1,j+rad)
    for k in range(disparityRange+1):            
        error = 0 
        for ii in range(minr,maxr):
            for jj in range(minc, maxc):
                sh_index_i = ii-i
                sh_index_j = jj-j
                error += abs(sh_leftimg[sh_i+sh_index_i,sh_j+sh_index_j] - d_rightimg[ii,jj+k+disparityRange])
                error += abs(d_gradientleft_x[ii,jj] - d_gradientright_x[ii,jj+k+disparityRange])
                error += abs(d_gradientleft_y[ii,jj] - d_gradientright_y[ii,jj+k+disparityRange])
                error += abs(d_Oleft[ii,jj] - d_Oright[ii,jj+k+disparityRange])


        d_disparity[i-rad,j-rad,k] = error





# ======================================================================
# wraper function
# ======================================================================
def pardisparity():
    '''
    wrapper function for computing the disparity matrix
    '''
    start = time.time()
    rightimg = np.array(Image.open('left_bw.png'),dtype=np.int32)
    leftimg  = np.array(Image.open('right_bw.png'),dtype=np.int32)
    rows,columns= leftimg.shape
    rad = RAD2

    #initialising 3D disparity matrix
    disparity = 255*np.ones(shape=(rows-2*rad,columns-2*rad,disparityRange+1))
    output = np.zeros(shape=(rows-2*rad,columns-2*rad))


    # padding the right image
    padding = 255*np.ones(shape=(rows,disparityRange)) 
    rightimg = np.append(padding,rightimg,axis = 1)           
    rightimg = np.append(rightimg,padding,axis = 1)
    gradientleft_x = 255.0*np.ones(shape=(rows,columns),dtype=np.float32)
    gradientleft_y = 255.0*np.ones(shape=(rows,columns),dtype=np.float32)
    gradientright_x = 255.0*np.ones(shape=(rows,columns+disparityRange),dtype=np.float32)
    gradientright_y = 255.0*np.ones(shape=(rows,columns+disparityRange),dtype=np.float32)

    Oleft  = 255.0*np.ones(shape=(rows,columns),dtype=np.float32)
    Oright = 255.0*np.ones(shape=(rows,columns+disparityRange),dtype=np.float32)



    d_leftimg  = cuda.to_device(leftimg)
    d_rightimg = cuda.to_device(rightimg)
    d_disparity  =  cuda.to_device(disparity)
    d_gradientleft_x = cuda.to_device(gradientleft_x)
    d_gradientleft_y = cuda.to_device(gradientleft_y)
    d_gradientright_x = cuda.to_device(gradientright_x)
    d_gradientright_y = cuda.to_device(gradientright_y)
    d_Oleft = cuda.to_device(Oleft)
    d_Oright = cuda.to_device(Oright)


    TPBx = TPB
    TPBy = TPB
    BPGx = (rows +TPBx-1)//TPBx
    BPGy = (columns +TPBy-1)//TPBy

    calc_gradientx[[BPGx,BPGy], [TPBx, TPBy]](d_gradientleft_x, d_leftimg, rows,columns)
    calc_gradienty[[BPGx,BPGy], [TPBx, TPBy]](d_gradientleft_y, d_leftimg, rows,columns)
    calc_gradientx[[BPGx,BPGy], [TPBx, TPBy]](d_gradientright_x, d_rightimg, rows,columns)
    calc_gradienty[[BPGx,BPGy], [TPBx, TPBy]](d_gradientright_y, d_rightimg, rows,columns)
    calc_orientation[[BPGx,BPGy], [TPBx, TPBy]](d_gradientleft_x, d_gradientleft_y,d_Oleft, rows,columns)
    calc_orientation[[BPGx,BPGy], [TPBx, TPBy]](d_gradientright_x, d_gradientright_y,d_Oright, rows,columns)



    kerneldisparity[[BPGx,BPGy], [TPBx, TPBy]](d_leftimg, d_rightimg , d_disparity ,d_gradientleft_x, d_gradientleft_y, d_gradientright_x, d_gradientright_y,d_Oleft,d_Oright  )
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
    plt.imshow(output[:,:350],cmap = "jet")
    plt.colorbar()
    plt.show()



# ======================================================================
# runing the model
# ======================================================================
disparityRange = 50
TPB = 16
RAD1 = 1
RAD2 = 2
SH_N = 18
SH_N2 = 20
pardisparity()
pardisparity()
pardisparity()




