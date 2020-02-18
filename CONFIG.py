'''Configure algorithm operation via this file'''
import os

# Path to set working directory
PATH = os.path.dirname(os.path.abspath( __file__ )) + "/"

# Input image name and extension
IMG = "lena"
EXT = ".png"

# Key paths
TEMP = "temp/"             # Folder used to store intermediary results
SRC  = "images/"           # Folder containing input and output

# FDIFF experiment path 
FDIFF_EXPERIMENT_PATH=PATH+"fDiff_Encrypt.txt"
FDIFF_DECRYPT_PATH=PATH+"fDiff_Decrypt.txt"
IMG_VECTOR_ENCRYPT_PATH=PATH+"imgVec_Encrypt.txt"
IMG_VECTOR_DECRYPT_PATH=PATH+"imgVec_Decrypt.txt"

# Input/Output images
ENC_IN =  SRC + IMG + EXT               # Input image
ENC_OUT= SRC + IMG + "_encrypted.png"   # Encrypted Image
DEC_OUT= SRC + IMG + "_decrypted.png"   # Decrypted Image
PERM   = TEMP + IMG + "_1permuted.png"    # Permuted Image
DIFF   = TEMP + IMG + "_2diffused.png"    # Diffused Image
UNDIFF = TEMP + IMG + "_3undiffused.png"  # UnDiffused Image
UNPERM = TEMP + IMG + "_4unpermuted.png"  # UnPermuted Image

# Flags
DEBUG_TIMER  = False # Print timing statistics in console
DEBUG_IMAGES = True # Store intermediary results
DEBUG_CONSTANTS=True # For x,y,a,b,c,offset
DEBUG_CONTROL_PARAMETERS=True  # For alpha and beta
RESIZE_TO_DEBUG =True # Make input image smaller to understand image contents
DEBUG_DIFFUSION=True # Print rDiff and fDiff
DEBUG_TRANSLATION=True# Print U and V 
DEBUG_SEQUENCES=True # Print P1 and P2
ENABLE_FDIFF_EXPERIMENT=False # Write fDiff during Encryption to file
ENABLE_IMG_VECTOR_EXPERIMENT=False # Write imgVector during Encryption and Decryption to file

# Image Resize Parameters
RESIZE_M=4
RESIZE_N=4

# Control Parameters
PERM_ROUNDS= 2
PERMINTLIM = 32
DIFFINTLIM = 16
f = 0
r = 0
MANTISSA_BITS=2