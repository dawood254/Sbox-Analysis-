import math
import numpy as np
import numpy
import Myfunctions
def Nonlinearity(Sbox):
    input_bits = int(math.log(len(Sbox),2))
    output_bits = int(math.log(max(Sbox)+1,2))
    print("Input Bits =",input_bits)
    print("Output Bits =",output_bits)
    Plain = list(range(0,pow(2,output_bits)))
    Bin_Plain = Myfunctions.Dec_to_Bin(Plain,output_bits)
    Bin_Plain = Bin_Plain[1:pow(2,output_bits),0:output_bits]
    Bin_Sbox = Myfunctions.Dec_to_Bin(Sbox,output_bits)
    Bin_Sbox = Myfunctions.Transpose(Bin_Sbox)
    Cor_fun = Myfunctions.Bin_Matrix_Mult(Bin_Plain,Bin_Sbox)
    TP_Sbox = Myfunctions.Polarity(Cor_fun)
    Hadamrd = Myfunctions.Hadamard(input_bits)
    Walsh = Myfunctions.Matrix_Mult(TP_Sbox,Hadamrd)
    Absolut = np.absolute(Walsh)
    N = pow(2,input_bits-1)-numpy.amax(Absolut,axis=1)/2
    size = len(N)
    for i in range(size):
        print("Nonlinearity of f_",i+1,'=',N[i])
def SAC(Sbox):
        Input_bits = int(math.log(len(Sbox),2))
        Output_bits = int(math.log(max(Sbox)+1,2))
        Plain = list(range(0,pow(2,Input_bits)))
        Data = list(range(0,Output_bits))
        AC = numpy.zeros((Output_bits,pow(2,Output_bits)-1))
        size = len(Data)
        for i in range(size):
                P1 = list(np.bitwise_xor(Plain,pow(2,Data[i])))
                alpha  = list(Myfunctions.Sub_bytes(P1,Sbox))
                out = np.bitwise_xor(alpha,Sbox)
                AC[i,:]=Myfunctions.Compute_Probability(out,Input_bits,Output_bits)
        return print(AC)

def Differential_Uniform(Sbox):
        Input_bits = int(math.log(len(Sbox),2))
        Output_bits = int(math.log2(max(Sbox)+1))
        Plain = list(range(0,pow(2,Input_bits)))
        DDT = numpy.zeros((pow(2,Input_bits),pow(2,Output_bits)))
        for i in range(pow(2,Input_bits)):
                 Alpha = np.bitwise_xor(Plain,i)
                 Sbox_Alpha = list(Myfunctions.Sub_bytes(Alpha,Sbox))
                 Diff = np.bitwise_xor(Sbox,Sbox_Alpha)
                 DDT[i,:]=Myfunctions.Count(Diff,Output_bits)
        return print(DDT)

def Linear_Approx(Sbox):
     Input_Bits = int(math.log(len(Sbox),2))
     Output_Bits = int(math.log(max(Sbox)+1,2))
     Plain = list(range(0,pow(Input_Bits,2)))
     Bin_Plain = Myfunctions.Dec_to_Bin(Plain,Input_Bits)
     Bin_Plain = Myfunctions.Transpose(Bin_Plain)
     Bin_Sbox = Myfunctions.Dec_to_Bin(Sbox,Output_Bits)
     Bin_Sbox = Myfunctions.Transpose(Bin_Sbox)
     LAT = numpy.zeros((pow(2,Input_Bits),pow(2,Output_Bits)))
     for i in range(pow(2,Input_Bits)):
         for j in range(pow(2,Output_Bits)):
             P=Myfunctions.Linear_Comb(Bin_Plain,Bin_Sbox,i,j,Input_Bits,Output_Bits)
             LAT[i,j]=P
     return print(LAT)