import sys
import os                   # Path setting and file-retrieval
import cv2                  # OpenCV
import time                 # Timing Execution
import numpy as np          # See above
import CONFIG as cfg        # Debug flags and constants
from shutil import rmtree   # Directory removal
import secrets              # CSPRNG
import warnings             # Ignore integer overflow during diffusion
import math                 # For floor()

warnings.filterwarnings("ignore", category=RuntimeWarning)
#Print a numpy array, no matter its size
#np.set_printoptions(threshold=sys.maxsize)

os.chdir(cfg.PATH)

# Path-check and image reading
def Init():
    if not os.path.exists(cfg.SRC):
        print("Input directory does not exist!")
        raise SystemExit(0)
    else:
        if os.path.isfile(cfg.ENC_OUT):
            os.remove(cfg.ENC_OUT)
        if os.path.isfile(cfg.DEC_OUT):
            os.remove(cfg.DEC_OUT)

    if os.path.exists(cfg.TEMP):
        rmtree(cfg.TEMP)
    os.makedirs(cfg.TEMP)

    # Open Image
    img = cv2.imread(cfg.ENC_IN,0)
    if cfg.RESIZE_TO_DEBUG==True:
        img=cv2.resize(img,(cfg.RESIZE_M,cfg.RESIZE_N),interpolation=cv2.INTER_LANCZOS4)
        print("\ninput image=\n")
        print(img)
    if img is None:
        print("File does not exist!")
        raise SystemExit(0)
    
    print("\nM="+str(img.shape[0])+"\nN="+str(img.shape[1]))
    return img, img.shape[0], img.shape[1]

# Generate and return rotation vector of length n containing values < m
def genRelocVec(m, n, logfile):
    # Initialize constants
    secGen = secrets.SystemRandom()
    a = secGen.randint(2,cfg.PERMINTLIM)
    b = secGen.randint(2,cfg.PERMINTLIM)
    c = 1 + a*b
    x = secGen.uniform(0.0001,1.0)
    y = secGen.uniform(0.0001,1.0)
    offset = secGen.randint(2,cfg.PERMINTLIM)

    # Log parameters for decryption
    with open(logfile, 'w+') as f:
        f.write(str(a) +"\n")
        f.write(str(b) +"\n")
        f.write(str(x) +"\n")
        f.write(str(y) +"\n")
        f.write(str(offset) + "\n")

    # Skip first <offset> values
    for i in range(offset):
        x = (x + a*y)%1 
        y = (b*x + c*y)%1
    
    # Start writing intermediate values
    ranF = np.zeros((m*n),dtype=float)
    for i in range(m*n//2):
        x = (x + a*y)%1
        y = (b*x + c*y)%1
        ranF[2*i] = x
        ranF[2*i+1] = y
        if x==0 or y==0:
            null = "null"
            return null, null
    
    # Generate column-relocation vector
    r = secGen.randint(1,m*n-n)
    exp = 10**14
    vec = np.zeros((n),dtype=int)
    for i in range(n):
        vec[i] = int((ranF[r+i]*exp)%m)

    with open(logfile, 'a+') as f:
        f.write(str(r))

    return ranF, vec

#Column rotation
def rotateColumn(img, col, colID, offset):
    colLen = len(col)
    for i in range(img.shape[0]): # For each row
        img[i][colID] = col[(i+offset)%colLen]
        
# Row rotation
def rotateRow(img, row, rowID, offset):
    rowLen = len(row)
    for j in range(img.shape[1]): # For each column
        img[rowID][j] = row[(j+offset)%rowLen]

#Write image to disk
def write_image(img,img_path,filename):
    final_path=img_path+filename
    cv2.imwrite(final_path,img)

#Displays a given image using pyplot
def display(img):       
    imgrgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(imgrgb)
    plt.show()

#Write the fDiff to file
def writefDiff(fDiff,filename):
    with open(filename,"w+") as f:
        for i in range(0,len(fDiff)):
            f.write(str(fDiff[i])+"\n")

#Write the imgVec to a file
def writeimgVec(imgVec,filename):
    with open(filename,"w+") as f:
        for i in range(0,len(imgVec)):
            f.write(str(imgVec[i])+"\n")         

#Tells us how many elements are  greater than 255
def count_greater_255(vec):
    count=0
    for i in range(len(vec)):
        if vec[i] > 255:
            count=count+1
            
    print("\ncount="+str(count))
    for i in range(len(vec)):
        if vec[i] > 255:
            vec[i]=vec[i]-255
    return vec        
            
def Encrypt():
    # Read image
    img, m, n = Init()

    # Col-rotation | len(U)=n, values from 0->m
    P1, U = genRelocVec(m,n,"temp/P1.txt") 
    while type(U) is str:
        P1, U = genRelocVec(m,n,"temp/P1.txt")

    # Row-rotation | len(V)=m, values from 0->n
    P2, V = genRelocVec(n,m,"temp/P2.txt") 
    
    while type(V) is str:
        P2, V = genRelocVec(n,m,"temp/P2.txt")

    
    if cfg.DEBUG_TRANSLATION==True:
        print("\nU=")
        print(U)
        print("\nV=")
        print(V)
    
    if cfg.DEBUG_SEQUENCES==True:
        print("\nP1=")
        print(P1)
        print("\nP2=")
        print(P2)    

    for i in range(cfg.PERM_ROUNDS):
        # For each column
        for j in range(n):
            if U[j]!=0:
                rotateColumn(img, np.copy(img[:,j]), j, U[j])
        
        if cfg.RESIZE_TO_DEBUG==True:
            print("\nColumn rotated image=\n")
            print(img)

        # For each row
        for i in range(m):
            if V[i]!=0:
                rotateRow(img, np.copy(img[i,:]), i, V[i])

        if cfg.RESIZE_TO_DEBUG==True:
            print("\nColumn and Row rotated image=\n")
            print(img)

    if cfg.DEBUG_IMAGES==True:
        cv2.imwrite(cfg.PERM, img)

    '''PERMUTATION PHASE COMPLETE'''

    # Convert image to grayscale and flatten it
    #imgBW = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    imgVec = np.asarray(img).reshape(-1)
    fDiff = np.zeros_like(imgVec)
    rDiff = np.zeros_like(imgVec)
    
    if cfg.RESIZE_TO_DEBUG==True:
        print("\nimgVec=\n")
        print(imgVec)

    if cfg.ENABLE_IMG_VECTOR_EXPERIMENT==True:
        writeimgVec(imgVec,cfg.IMG_VECTOR_ENCRYPT_PATH)    
    
    # Initiliaze diffusion constants
    secGen = secrets.SystemRandom()
    alpha = secGen.randint(1,cfg.DIFFINTLIM)
    beta = secGen.randint(1,cfg.DIFFINTLIM)
    mn = len(imgVec)
    mid = mn//2
    f, r = cfg.f, cfg.r
    
    # Forward Diffusion
    j=0
    fDiff[0] = f + imgVec[0] + alpha*(P1[0] if f&1==0 else P1[1]) # 0
    # 1->(mid-1)
    for i in range(1, mid):
        fDiff[i] = fDiff[i-1] + imgVec[i] + alpha*(P1[2*i] if fDiff[i-1]&1==0 else P1[2*i + 1])
    # mid->(mn-1)
    for i in range(mid, mn): 
        fDiff[i] = fDiff[i-1] + imgVec[i] + alpha*(P1[2*j] if fDiff[i-1]&1==0 else P1[2*j + 1])
        j += 1

    if cfg.ENABLE_FDIFF_EXPERIMENT==True:    
        writefDiff(fDiff,cfg.FDIFF_EXPERIMENT_PATH)    
    
    # Reverse Diffusion
    rDiff[mn-1] = r + fDiff[mn-1] + beta*(P2[mn-2] if r&1==0 else P2[mn-1]) # (mn-1)
    
    j = mid-1
    # (mn-2)->mid
    for i in range(mn-2, mid-1, -1): 
        rDiff[i] = rDiff[i+1] + fDiff[i] + beta*(P2[2*j] if rDiff[i+1]&1==0 else P2[2*j + 1])
        j -= 1
    # (mid-1)->0
    for i in range(mid-1, -1, -1): 
        rDiff[i] = rDiff[i+1] + fDiff[i] + beta*(P2[2*i] if rDiff[i+1]&1==0 else P2[2*i + 1])

    #fDiff=count_greater_255(fDiff)
    #rDiff=count_greater_255(rDiff)   
    
    if cfg.DEBUG_DIFFUSION==True:
        print("\nfDiff=")
        print(fDiff)
        print("\nrDIff=")
        print(rDiff) 

    # Log diffusion parameters for decryption
    with open("temp/diff.txt","w+") as f:
        f.write(str(alpha) + "\n")
        f.write(str(beta) + "\n")

    img = (np.reshape(rDiff,img.shape)).astype(np.uint8)

    if cfg.DEBUG_IMAGES==True:
        cv2.imwrite(cfg.DIFF, img)

    '''DIFFUSION PHASE COMPLETE'''
    
    cv2.imwrite(cfg.ENC_OUT, img)

Encrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()