import os                   # Path setting and file-retrieval
import cv2                  # OpenCV
import time                 # Timing Execution
import numpy as np          # See above
import CONFIG as cfg        # Debug flags and constants
from shutil import rmtree   # Directory removal
import secrets              # CSPRNG
import warnings             # Ignore integer overflow during diffusion
from pyfloat import get_n_mantissa_bits # For P1 and P2

warnings.filterwarnings("ignore", category=RuntimeWarning)

os.chdir(cfg.PATH)

# Path-check and image reading
def Init():
    if not os.path.exists(cfg.SRC):
        print("Input directory does not exist!")
        raise SystemExit(0)

    if not os.path.exists(cfg.TEMP):
        print("Error! Decryption parameters not found!")
        raise SystemExit(0)

    # Open Image
    img = cv2.imread(cfg.ENC_OUT,-1)
    if img is None:
        print("File does not exist!")
        raise SystemExit(0)
    return img, img.shape[0], img.shape[1]

# Generate and return rotation vector of length n containing values < m
def genRelocVec(m, n, logfile):
    # Read constants from logfile
    f = open(logfile, "r")
    fl = f.readlines()
    f.close()
    a = int(fl[0])
    b = int(fl[1])
    c = 1 + a*b
    x = float(fl[2])
    y = float(fl[3])
    offset = int(fl[4])

    if cfg.DEBUG_CONSTANTS==True:
        print("\na="+str(a))
        print("\nb="+str(b))
        print("\nc="+str(c))
        print("\nx="+str(x))
        print("\ny="+str(y))
        print("\noffset="+str(offset))

    # Skip first offset values
    for i in range(offset):
        x = (x + a*y)%1
        y = (b*x + c*y)%1
    

    if cfg.DEBUG_CONSTANTS==True:
        print("\nFinal x="+str(x))
        print("\nFinal y="+str(y))


    # Start writing intermediate values
    ranF = np.zeros((m*n),dtype=float)
    for i in range(m*n//2):
        x = (x + a*y)%1
        y = (b*x + c*y)%1
        ranF[2*i] = x
        ranF[2*i+1] = y


    ranFInt=np.zeros((m*n),dtype=np.uint8)
    for i in range(m*n//2):
        ranFInt[2*i]=get_n_mantissa_bits(ranF[2*i],cfg.MANTISSA_BITS)    
        ranFInt[2*i+1]=get_n_mantissa_bits(ranF[2*i+1],cfg.MANTISSA_BITS)
    
    # Generate column-relocation vector
    r = int(fl[5])
    exp = 10**14
    vec = np.zeros((n),dtype=int)
    for i in range(n):
        vec[i] = int((ranF[r+i]*exp)%m)
    return ranFInt, vec

# Column rotation
def rotateColumn(img, col, colID, offset):
    colLen = len(col)
    for i in range(img.shape[0]): # For each row
        img[i][colID] = col[(i+offset)%colLen]
        
# Row rotation
def rotateRow(img, row, rowID, offset):
    rowLen = len(row)
    for j in range(img.shape[1]): # For each column
        img[rowID][j] = row[(j+offset)%rowLen]

def Decrypt():
    # Read image
    print("\nIn RBD\n")
    img, m, n = Init()

    # Generate rotation vectors w/ CSPRNG
    P1, U = genRelocVec(m,n,"temp/P1.txt") # Col-rotation | len(U)=n, values from 0->m
    P2, V = genRelocVec(n,m,"temp/P2.txt") # Row-rotation | len(V)=m, values from 0->n
    
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

    # Read diffusion parameters
    f = open("temp/diff.txt","r")
    alpha = int(f.readline())
    beta = int(f.readline())
    f.close()

    if cfg.DEBUG_CONTROL_PARAMETERS==True:
        print("\nalpha on retrieval="+str(alpha))
        print("\nbeta on retrieval="+str(beta))
    
    # Flatten image to image Reverse-Diffused vector
    rDiff = np.asarray(img).reshape(-1)
    fDiff = np.empty_like(rDiff)
    imgVec = np.empty_like(rDiff)
    mn = len(rDiff)
    mid = mn//2
    f, r = 0, 0

    print("\nREGENERATE FDIFF 1st PHASE\n")
    # Regenerate fDiff[]
    for i in range(0, mid):
        fDiff[i] = rDiff[i] - rDiff[i+1] - beta*(P2[2*i] if rDiff[i+1]&1==0 else P2[2*i + 1])

        if rDiff[i+1]&1==0:
            print("\ni="+str(i))
            print("\nfDiff[i]="+str(fDiff[i])+"\trDiff[i]="+str(rDiff[i])+"\trDiff[i+1]="+str(rDiff[i+1])+"\tP2[2*i]="+str(P2[2*i]))
        else:
            print("\ni="+str(i))
            print("\nfDiff[i]="+str(fDiff[i])+"\trDiff[i]="+str(rDiff[i])+"\trDiff[i+1]="+str(rDiff[i+1])+"\tP2[2*i+1]="+str(P2[2*i+1]))

    print("\nREGENERATE FDIFF 2nd PHASE\n")
    j = 0
    print("\nj before starting REGENERATE FDIFF 2nd PHASE="+str(j))
    for i in range(mid, mn-1):
        fDiff[i] = rDiff[i] - rDiff[i+1] - beta*(P2[2*j] if rDiff[i+1]&1==0 else P2[2*j + 1])
        
        if rDiff[i+1]&1==0:
            print("\ni="+str(i))
            print("\nj="+str(j))
            print("\nfDiff[i]="+str(fDiff[i])+"\trDiff[i]="+str(rDiff[i])+"\trDiff[i+1]="+str(rDiff[i+1])+"\tP2[2*j]="+str(P2[2*j]))
        else:
            print("\ni="+str(i))
            print("\nj="+str(j))
            print("\nfDiff[i]="+str(fDiff[i])+"\trDiff[i]="+str(rDiff[i])+"\trDiff[i+1]="+str(rDiff[i+1])+"\tP2[2*j+1]="+str(P2[2*j+1]))

        j += 1

    fDiff[mn-1] = rDiff[mn-1] - r  - beta*(P2[mn-2] if r&1==0 else P2[mn-1])

    if r&1==0:
        print("\nfDiff[mn-1]="+str(fDiff[mn-1])+"\trDiff[mn-1]="+str(rDiff[mn-1])+"\tr="+str(r)+"\t"+"\tP2[mn-2]="+str(P2[mn-2]))
    else:
        print("\nfDiff[mn-1]="+str(fDiff[mn-1])+"\trDiff[mn-1]="+str(rDiff[mn-1])+"\tr="+str(r)+"\t"+"\tP2[mn-1]="+str(P2[mn-1]))

    print("\nREGENERATE IMGVEC\n")

    # Regenerate imgVec[]
    print("\nj before starting REGENERATE IMGVEC=\n"+str(j))
    j = mid-1
    for i in range(mn-1, mid-1, -1):
        imgVec[i] = fDiff[i] - fDiff[i-1] - alpha*(P1[2*j] if fDiff[i-1]&1==0 else P1[2*j + 1])
        
        if fDiff[i-1]==0:
            print("\ni="+str(i))
            print("\nj="+str(j))
            print("\nimgVec[i]="+str(imgVec[i])+"\tfDiff[i]="+str(fDiff[i])+"\tfDiff[i-1]="+str(fDiff[i-1])+"\tP1[2*j]="+str(P1[2*j]))
        else:
            print("\ni="+str(i))
            print("\nj="+str(j))
            print("\nimgVec[i]="+str(imgVec[i])+"\tfDiff[i]="+str(fDiff[i])+"\tfDiff[i-1]="+str(fDiff[i-1])+"\tP1[2*j+1]="+str(P1[2*j+1]))   

        j -= 1
    
    for i in range(mid-1, -1, -1):
        imgVec[i] = fDiff[i] - fDiff[i-1] - alpha*(P1[2*i] if fDiff[i-1]&1==0 else P1[2*i + 1])

        if fDiff[i-1]==0:
            print("\ni="+str(i))
   
            print("\nimgVec[i]="+str(imgVec[i])+"\tfDiff[i]="+str(fDiff[i])+"\tfDiff[i-1]="+str(fDiff[i-1])+"\tP1[2*i]="+str(P1[2*i]))
        else:
            print("\ni="+str(i))
            print("\nimgVec[i]="+str(imgVec[i])+"\tfDiff[i]="+str(fDiff[i])+"\tfDiff[i-1]="+str(fDiff[i-1])+"\tP1[2*i+1]="+str(P1[2*i+1]))  

    imgVec[0] = fDiff[0] - f - alpha*(P1[0] if f&1==0 else P1[1])

    if f&1==0:
        print("\nimgVec[0]="+str(imgVec[0])+"\tfDiff[0]="+str(fDiff[0])+"\tf="+str(f)+"\tP1[0]="+str(P1[0]))
    else:
        print("\nimgVec[0]="+str(imgVec[0])+"\tfDiff[0]="+str(fDiff[0])+"\tf="+str(f)+"\tP1[1]="+str(P1[1]))    

    if cfg.RESIZE_TO_DEBUG==True:
        print("\nimgVec=")
        print(imgVec)
        print("\nrDiff=")
        print(rDiff)
        print("\nfDiff=")
        print(fDiff)

    # Reshape into matrix
    img = (np.reshape(imgVec,img.shape)).astype(np.uint8)

    print("\nDecrypted image=\n")
    print(img)

    if cfg.DEBUG_IMAGES==True:
        cv2.imwrite(cfg.UNDIFF, img)
    
    '''DIFFUSION PHASE COMPLETE'''
    
    for i in range(cfg.PERM_ROUNDS):
        # For each row
        for i in range(m):
            if V[i]!=0:
                rotateRow(img, np.copy(img[i,:]), i, n-V[i])
        
        
        # For each column
        for j in range(n):
            if U[j]!=0:
                rotateColumn(img, np.copy(img[:,j]), j, m-U[j])
    
    print("\nDecrypted Image=\n")
    print(img)

    if cfg.DEBUG_IMAGES:
        cv2.imwrite(cfg.UNPERM,img)

    '''PERMUTATION PHASE COMPLETE'''
    
    cv2.imwrite(cfg.DEC_OUT, img)

Decrypt()
cv2.waitKey(0)
cv2.destroyAllWindows()