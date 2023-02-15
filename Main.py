import math
import numpy as np
import numpy
import Myfunctions
import Sbox_Analyses
import cv2
Sbox = [0,1,9,14,13,11,7,6,15,2,12,5,10,4,3,8]
#Sbox = [0,1,2,13,4,7,15,6,8,12,5,3,10,14,11,9,0,1,2,13,4,7,15,6,8,12,14,11,10,9,3,5,0,1,2,13,4,7,15,6,8,14,11,10,5,9,12,3,0,1,2,13,4,7,15,6,8,14,12,11,3,9,5,10]
Sbox_Analyses.Nonlinearity(Sbox)
Sbox_Analyses.SAC(Sbox)
Sbox_Analyses.Differential_Uniform(Sbox)
Sbox_Analyses.Linear_Approx(Sbox)