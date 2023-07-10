import math
import numpy as np
import numpy
def Dec_to_Bin(Sbox,n): 
    Bin_Sbox = numpy.zeros((len(Sbox),n))
    for i in range(len(Sbox)):
       Bin_Sbox[i,:]=list(map(int,list(bin(Sbox[i])[2: ].zfill(n))))
    return Bin_Sbox

def Polarity(B_Sbox):
    x = len(B_Sbox[:,0])
    y = len(B_Sbox[0,:])
    P_Sbox = numpy.zeros((x,y))
    for i in range(x):
        R = B_Sbox[i,:]
        for j in range(y):
           P_Sbox[i,j]=pow(-1,R[j])
    return P_Sbox

def dot(A,B):
 s = len(A)
 c=0
 for i in range(s):
     c = c+A[i]*B[i]
 return c 

def Matrix_Mult(A,B):
 ar = np.shape(A)[0]
 ac = np.shape(A)[1]
 br = np.shape(B)[0]
 bc = np.shape(B)[1]
 O = numpy.zeros((ar,bc))
 if (ac==br):
     for i in range(ar):
        C = A[i,:]
        for j in range(bc):
         C1 = B[:,j]
         O[i,j]=dot(C,C1)
 else:
  print("The number of row and the number of columns are not equal")
 return O

def Bin_Matrix_Mult(A,B):
    ar = np.shape(A)[0]
    ac = np.shape(A)[1]
    br = np.shape(B)[0]
    bc = np.shape(B)[1]
    O = numpy.zeros((ar,bc))
    if (ac==br):
         for i in range(ar):
             if ar==1:
                 C = A[0]
             else:
                 C = A[i,:]
         for j in range(bc):
             if bc==1:
                  C1 = B[0]
                  O[i,j]=dot(C,C1)%2
             else:
                  C1 = B[:,j]
                  O[i,j]=dot(C,C1)%2
        
    else:
     print("The number of row and the number of columns are not equal")
    return O

def Hadamard(N):
    A = np.array([[1]])
    for i in range(N):
        B = np.append(A,A,axis=1)
        C = np.append(A,-1*A,axis=1)
        A = np.concatenate((B,C))
    return A
def Transpose(A):
    row =  len(A[:,0])
    col  = len(A[0,:])
    T = numpy.zeros((col,row)) 
    for i in range(row):
        for j in range(col):
            T[j,i]=A[i,j]
    return T
def Sub_bytes(P,Sbox):
    s = len(P)
    Sub = numpy.zeros((s))
    print(Sub[0])
    for i in range(s):
        Sub[i] = Sbox[P[i]]
    return Sub.astype(int)
def Sum_Probability(Sbox,Input_bits,output_bits):
     Plain = list(range(0,pow(2,output_bits)))
     Bin_Plain = Dec_to_Bin(Plain,output_bits)
     Bin_Sbox = Dec_to_Bin(Sbox,output_bits)
     Bin_Sbox = Transpose(Bin_Sbox)
     Cor_fun =  Bin_Matrix_Mult(Bin_Plain,Bin_Sbox)
     size = len(Cor_fun[0,:])
     Probab = numpy.zeros((size))
     for i in range(size):
        Probab[i] = sum(Cor_fun[:,i])
     return Probab

def Compute_Probability(out,Input_bits,Output_bits):
    Plain1 = list(range(1,pow(2,Output_bits)))
    Bin_Plain1 = Dec_to_Bin(Plain1,Output_bits)
    Bin_Out =  Dec_to_Bin(out,Output_bits)
    Bin_Out =  Transpose(Bin_Out)
    Cor_fun =  Bin_Matrix_Mult(Bin_Plain1,Bin_Out)
    size = len(Cor_fun[:,0])
    Probab = numpy.zeros((size))
    for i in range(size):
        Probab[i] = sum(Cor_fun[i,:])
    return Probab/pow(2,Input_bits)

def Count(X,n):
        size = len(X)
        S = pow(2,n)
        C = numpy.zeros((S))
        i = 0
        while (i<S):
                j = 0
                k = 0
                while (k<size):
                        if X[k] == i:
                                j=j+1
                                k=k+1
                        else:
                                k=k+1
                C[i] = j
                i=i+1
        return C
    
def Linear_Comb(Bin_Plain,Bin_Sbox,x,y,Input_Bits,Output_Bits):
     Alpha = [list(map(int,list(bin(x)[2: ].zfill(Input_Bits)))),]
     Beta = [list(map(int,list(bin(2)[2: ].zfill(Output_Bits)))),]
     Alpha_Plain = Bin_Matrix_Mult(Alpha,Bin_Plain)
     Alpha_Plain = Alpha_Plain[0]
     Beta_Sbox = Bin_Matrix_Mult(Beta,Bin_Sbox)
     Beta_Sbox = Beta_Sbox[0]
     y = len(Beta_Sbox)
     count=0
     for i in range(y):
         if Alpha_Plain[i]==Beta_Sbox[i]:
             count=count+1
     return count