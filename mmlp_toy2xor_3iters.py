


"""

Toy 2 : XOR
  
  

Author: Nathan E. Frick
 2020

in the multiplet neural network
 backprop

*** static b and variable m within each multiplet ***

"""


b_alpha = 0.0
m_alpha = 0.4
m_alpha2 = 0.2
w_alpha= 0.01
guseCaseSlope = 0

import math
import time
import numpy as np
#import pandas
#import pickle,gzip

np.set_printoptions(precision=3,suppress=True)

import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d

#rng = np.random.default_rng(12345)
# print("=== shuffle random generator =====")
tmpR = np.random.laplace(0.5,0.3)
tmpR = np.random.laplace(0.5,0.3)
tmpR = np.random.laplace(0.5,0.3)
tmpR = np.random.laplace(0.5,0.3)
tmpR = np.random.laplace(0.5,0.3)
tmpR = np.random.laplace(0.5,0.3)
# print("=== end  =====")
# print(" ")

#fig = plt.figure()
#ax = plt.axes(projection='3d')

#fig.savefig("1.png",bbox_inches="tight")

"""
        self.alphaM = 0.1
        self.alphaB = 0.1
        self.alphaW = 0.1
"""



#  multiplet neuron layer class


class mmlp_layer():
# holds pointers to layer neurons 
#  and numpy format output array
    def __init__(self):
        self.mn = []   # multiplet neuron array
        self.wtgrp = []
        self.vecOutI = np.array([])  # numpy format layer output vector generated from self.mn array on forward pass
                                     # also used for possible storage of input vector instance, if layer 0
        self.categoryTargetI = np.array([]) # cache used to store network target instance (supervised learning, of course)                              

    def hardsetVecOut(self,theVector):
        self.vecOutI = np.ravel(theVector)

    def hardsetCategoryTargets(self,ctargetAry):
        self.categoryTargetI = np.ravel(ctargetAry)
    
    def clearVecOut(self):
        self.vecOutI = np.array([])  # almost certainly not the best way  (this is for testing only)

    def appendVecOut(self,ae1):
        self.vecOutI = np.append(self.vecOutI,ae1)   # note: this is slow and for testing only 



# makes a multiplet input vector stack, while precalculating powers

class multiplet_vec(): 
    def __init__(self,xs=[]):
        self.near_zero = 0.0001
        
#        dta = np.ravel(xs)
        dta = np.ravel(xs)
        self.x = np.stack((dta,dta,dta,dta,dta,dta,dta,dta,dta,dta,dta,dta,dta,dta,dta))
        self.cache_powers(dta)

    def cache_powers(self,newX): 
# fills in (for preallocated np array) the input vector stack (rows of powers of base vector) matrix
# integer powers from (-5 to 9) 
# the first row is the base vector to the minus 5 power (powerIndexOffset)
# x will be a numpy 2d array where the seventh row is the base vector (of power 1) 
        self.powerIndexOffset = 5   
        newX[newX == 0] = self.near_zero  # replace zero values with small number
        self.x[5] = 1.0
        np.copyto(self.x[6],newX)
        self.x[4] = 1.0 / self.x[6]
        self.x[7] = self.x[6] * self.x[6]   #squared
        self.x[8] = self.x[6] * self.x[7]   #^3
        self.x[9] = self.x[7] * self.x[7]   #^4
        self.x[10] = self.x[7] * self.x[8]  #^5
        self.x[11] = self.x[8] * self.x[8]  #^6
        self.x[12] = self.x[8] * self.x[9]  #^7
        self.x[13] = self.x[9] * self.x[9]  #^8
        self.x[14] = self.x[9] * self.x[10]  #^9
        self.x[3] = self.x[4] * self.x[4]  #^-2
        self.x[2] = self.x[4] * self.x[3]  #^-3
        self.x[1] = self.x[3] * self.x[3]  #^-4
        self.x[0] = self.x[3] * self.x[2]  #^-5

    def setval(self,xs=[]):
        self.cache_powers(np.ravel(xs))
        
        

class multiplet_weight_vec():
# this works with above input vector/tensor
    def __init__(self,w1):
        self.w = np.ravel(w1)
        self.errSum = 0.0
        self.ddw = 0.0

    def setval(self,w1):
        self.w = np.ravel(w1)

    def accumErr(self,val2a):
        self.errSum += val2a

    def clearErrSum(self):
        self.errSum = 0.0 
        self.ddw = 0.0

    def chkWtContraints(self):
        cwcminVal = np.amin(self.w)
        cwcmaxVal = np.amax(self.w)
        if (cwcminVal < 0.0):
            self.w += np.abs(cwcminVal) 
        if (cwcmaxVal > 1000000.0):
            self.w /= cwcmaxVal;

    def applyWDeltas(self,w_nu,useCaseSlope=0,cSlpScore=1.0):
        ddE = self.errSum
        if (useCaseSlope == 1):
            self.w -= ((w_nu * cSlpScore) * ddE) * self.ddw   #  * actTerm
        else:
            self.w -= (w_nu * ddE) * self.ddw  # * actTerm
        self.chkWtContraints()    




    
class multiplet_neuron():
    def __init__(self,wPtr,p=1,q=1,m=1.0,b=0.0):
        self.p = p
        self.m = m
        self.q = q
        self.p_q = (p - q)
        self.b = b
        self.wPtr = wPtr
        self.numer = 1.0
        self.denom = 1.0
        self.cSlpScore = 1.0
        self.ddb = 0.0
        self.ddm = 0.0
        self.ddw = 0.0
        self.deltaE = 0.0
        self.delEW = 0.0
        


    def calcOut(self,xv):    
# calc the numerator and denominator sigmas 
#        print(xv.x[self.p+5])
#        print(xv.x[self.p_q+5])
        self.numer = np.vdot(self.wPtr.w,xv.x[self.p+5])
        self.denom = np.vdot(self.wPtr.w,xv.x[self.p_q+5])
#        print(self.numer)
#        print(self.denom)
        if (self.denom != 0.0):
            self.aout =  self.m * self.numer / self.denom + self.b
        else:
            self.aout = 0.0
        # create cache a vector for later (?)
        # self.denom_x = self.denom * xv.x[self.p+5]
        # self.numer_x = self.numer * xv.x[self.p_q+5]     

    def calcPartialDers(self,xv):
        # partial der with respect to b
        self.ddb = 1.0
        # partial der with respect to m, then w
        if (self.denom != 0.0):
            self.ddm = self.numer / self.denom
            m_denom2 = (self.m  / (self.denom * self.denom))
            # see paper: Informal Intro to Multiplet Neural Networks, N.E. Frick
            # with repect to w_k, but for all k - so a vector 
            # vector ddw = scalar * (vector - vector)
            self.ddw = m_denom2 * (self.denom * xv.x[self.p+5] - self.numer * xv.x[self.p_q+5])
            # this partial is not in the first preprint - multiply is element-by-element multiply,
            #  (since the derivative with respect to x_k includes a w_k term)
            #  this ddx will be used by the previous layer (during backprop)
            self.ddx =  m_denom2 * (  (self.denom * self.p) * np.multiply(self.wPtr.w,xv.x[(self.p-1)+5])
                - ((self.numer * self.p_q) * np.multiply(self.wPtr.w,xv.x[(self.p_q-1)+5])))


    def applyPtLearnNoActvtn(self,b_nu,m_nu):
        ddE = self.deltaE
        self.delEW = self.wPtr.errSum
        #  this version - static b
        self.b -= b_nu * self.ddb * ddE   # * actTerm
        self.m -= m_nu * self.ddm * ddE   # * actTerm
        
    
    def clearTotalE(self):
        self.deltaE = 0.0


    def accumTotalE(self,partial_dEdout):
        self.deltaE += partial_dEdout
        self.wPtr.ddw += self.ddw  # weight error partials are accumulated within multiplet_weight_vec class




def forwardPassInstance(inputVector,targets):
    netLyr[0].hardsetVecOut(inputVector)
    netLyr[0].hardsetCategoryTargets(targets)
    for l in range(1,len(netLyr)): # start with layer 1

        print('------- layer ------ ',l,' --------')

        xAry = netLyr[l-1].vecOutI
        yOut = netLyr[l-1].categoryTargetI
        xAryj = multiplet_vec(xAry)

        for lk1 in range(len(netLyr[l].mn)):  # traverse through layer neurons
            netLyr[l].mn[lk1].calcOut(xAryj)  # forward prop
            # this step is not in normal Multlayer Perceptrons, since it is such a straightforward deriviative
            # ( no activations here)
            netLyr[l].mn[lk1].calcPartialDers(xAryj)  

            # print('ddx is = ',netLyr[l].mn[lk1].ddx)
            #print('netLyr[l].mn lk1=',lk1,' aout=',netLyr[l].mn[lk1].aout,' on p=',netLyr[l].mn[lk1].p,' q=',netLyr[l].mn[lk1].q)

        netLyr[l].hardsetCategoryTargets(yOut)  # pass forward the final desired output (so that it arrives at last layer)
        # reformat output vector so next layer can easily use
        #print('yOut is ',yOut,' layer categoryTargetI=',netLyr[l].categoryTargetI)
        netLyr[l].clearVecOut()
        for lk1 in range(len(netLyr[l].mn)):
            netLyr[l].appendVecOut(netLyr[l].mn[lk1].aout)
            #  format print output - a monitor point
            out1 = 'aout=' + "{0:.2f}".format(netLyr[l].mn[lk1].aout)
            out_p = "{:n}".format(netLyr[l].mn[lk1].p)
            out_lk1 = "{:n}".format(lk1)
            out_q = "{:n}".format(netLyr[l].mn[lk1].q)
            out_b = "{0:.2f}".format(netLyr[l].mn[lk1].b)
            out_m = "{0:.2f}".format(netLyr[l].mn[lk1].m)
            #'q=' + out_q +
            print('neuron ' + out_lk1 + ' ' + out1 + ' on p=' + out_p +  ' b=' + out_b + ' m=' + out_m)
            print('      w=',netLyr[l].mn[lk1].wPtr.w) 
            

def backpropInstance(inputVector,targets): 
    activationPartial = 1.0

# the w_i total error is accumulated within the weight_vec class, but idk if it really matters, since it is an error sum anyway

       # output layer
    print(targets)

    l = len(netLyr)-1
    print(len(netLyr[l].mn),' neurons in output layer')
    for lk1 in range(len(netLyr[l].mn)):  # output layer neurons
        netLyr[l].mn[lk1].clearTotalE() 
    for wk1 in range(len(netLyr[l].wtgrp)):    
        netLyr[l].wtgrp[wk1].clearErrSum()   # the weights w have a separate variable, since shared
    for lk1 in range(len(netLyr[l].mn)):  # output layer neurons
        ddxout = (netLyr[l].mn[lk1].aout - targets[lk1]) #derviative of MSE
        # print('ddxout is ',ddxout)
        errVal = ddxout * activationPartial
        netLyr[l].mn[lk1].wPtr.accumErr(errVal)
        netLyr[l].mn[lk1].accumTotalE(errVal)   # sum of d(Etotal) /  d Out
        # print('ddE is (output layer) ',ddxout)  
        # go ahead and calculate corrections
        netLyr[l].mn[lk1].applyPtLearnNoActvtn(b_alpha,m_alpha)
    for wk1 in range(len(netLyr[l].wtgrp)):
        netLyr[l].wtgrp[wk1].applyWDeltas(w_alpha)

        # hidden layers 

    for l in range(1,len(netLyr)-1): # up to (not including) output layer
        for lk1 in range(len(netLyr[l].mn)):  # current layer
            netLyr[l].mn[lk1].clearTotalE()    # must use this loop, since we have multiplets
        for wk1 in range(len(netLyr[l].wtgrp)):    
            netLyr[l].wtgrp[wk1].clearErrSum()   # the weights w have a separate variable, since shared

        for lk1 in range(len(netLyr[l].mn)):  # current layer
            # first, sum the error terms for multiplet weights
            for lk2 in range(len(netLyr[l+1].mn)):  # from subsequent layer
                errVal = netLyr[l+1].mn[lk2].ddx[lk1] * activationPartial * netLyr[l+1].mn[lk2].delEW  #sum star subsq
                netLyr[l].mn[lk1].wPtr.accumErr(errVal) # accumErr for weights, yes here - as wPtr (in neuron loop)
            for lk2 in range(len(netLyr[l+1].mn)):  # from subsequent layer
                errVal = netLyr[l+1].mn[lk2].ddx[lk1] * activationPartial * netLyr[l+1].mn[lk2].deltaE #sum star subsq
                netLyr[l].mn[lk1].accumTotalE(errVal)   # sum of d(Etotal) /  d Out
            netLyr[l].mn[lk1].applyPtLearnNoActvtn(b_alpha,m_alpha2)  
        for wk1 in range(len(netLyr[l].wtgrp)): # current layer
            netLyr[l].wtgrp[wk1].applyWDeltas(w_alpha)
                




"""

  G L O B A L S 

"""

"""

██████╗  █████╗ ████████╗ █████╗ 
██╔══██╗██╔══██╗╚══██╔══╝██╔══██╗
██║  ██║███████║   ██║   ███████║
██║  ██║██╔══██║   ██║   ██╔══██║
██████╔╝██║  ██║   ██║   ██║  ██║
╚═════╝ ╚═╝  ╚═╝   ╚═╝   ╚═╝  ╚═╝
                                 
"""
        
# supervisory input

# category array

yTargets = []

xAryIn = []
#xAryIn.append(x1)
# xAryIn.append([0.99,0.1,0.1])
# xAryIn.append([0.99,1.0,0.1])

xAryIn.append([0.99,0.99])
xAryIn.append([0.01,0.01])
xAryIn.append([0.99,0.01])
xAryIn.append([0.01,0.99])

yTargets.append([1.0,0.0])
yTargets.append([1.0,0.0])
yTargets.append([0.0,1.0])
yTargets.append([0.0,1.0])





"""

    Define layer structure 

"""

netLyr = []
netLyr.append(mmlp_layer())  # Layer 0 holds the input dataset
# define the input vector size for next layer
for i in range(len(xAryIn[0])):
    netLyr[0].mn.append(xAryIn[0][i])
netLyr[0].hardsetVecOut(xAryIn[0]) # this is the same as above loop
netLyr[0].hardsetCategoryTargets(yTargets[0]) # this is the same as above loop

# for k in range(len(yTargets)):
#     print(yTargets[k])

netLyr.append(mmlp_layer())  # Layer 1
for l in range(1,2): 
    vecSize = len(netLyr[l-1].mn)  # number of neurons in previous layer
    #  define layer weight groups
    # initialize the membership selection weights w_i to 1.0
    # weight groups correspond to neuron multiplets, since multiplets share weights w_i
    netLyr[l].wtgrp.append(multiplet_weight_vec(np.ones((vecSize,))))  
    # now attach neurons associated with these w groups
    i = 0
    #netLyr[l].mn.append(multiplet_neuron(netLyr[l].wtgrp[i],-3,1,np.random.laplace(0.5,0.3),0.0)) # "minimum-ish" ( error results with -4 or -5 - don't use)
    #netLyr[l].mn.append(multiplet_neuron(netLyr[l].wtgrp[i],1,1,np.random.laplace(0.5,0.3),0.0))  # "dot product"
    netLyr[l].mn.append(multiplet_neuron(netLyr[l].wtgrp[i],5,1,1,0.0)) # "max-ish"

    #netLyr[l].wtgrp.append(multiplet_weight_vec(np.ones((vecSize,))))  
    #i = i + 1
    # b = 1.0
    netLyr[l].mn.append(multiplet_neuron(netLyr[l].wtgrp[i],-3,1,(-1*1),1.0)) # "minimum-ish" ( error results with -4 or -5 - don't use)
    #netLyr[l].mn.append(multiplet_neuron(netLyr[l].wtgrp[i],1,1,(-1*np.random.laplace(0.5,0.3)),1.0))  # "dot product"
    #netLyr[l].mn.append(multiplet_neuron(netLyr[l].wtgrp[i],5,1,(-1*np.random.laplace(0.5,0.3)),1.0)) # "max-ish"
 # weight group for q=2
 #   netLyr[l].wtgrp.append(multiplet_weight_vec(np.ones((vecSize,))*1.0)) 
 #   netLyr[l].mn.append(multiplet_neuron(netLyr[l].wtgrp[1],1,2,np.random.laplace(0.5,0.3),0.0))  #  "multiply-ish"              
 # weight group for q=-1
 #   netLyr[l].wtgrp.append(multiplet_weight_vec(np.ones((vecSize,)))) 
 #   netLyr[l].mn.append(multiplet_neuron(netLyr[l].wtgrp[2],1,-1,np.random.laplace(0.5,0.3),0.0)) # "inverse-ish"
 # weight group for q=-2
 #   netLyr[l].wtgrp.append(multiplet_weight_vec(np.ones((vecSize,)))) 
 #   netLyr[l].mn.append(multiplet_neuron(netLyr[l].wtgrp[2],-1,-2,np.random.laplace(0.5,0.3),0.0)) # "inverse-ish"
    
#    for i in range(len(netLyr[l].wtgrp)):
#        netLyr[l].mn.append(multiplet_neuron(netLyr[l].wtgrp[i],-3,1,np.random.laplace(0.5,0.3),0.0)) # "minimum-ish"
#        netLyr[l].mn.append(multiplet_neuron(netLyr[l].wtgrp[i],-1,-2,np.random.laplace(0.5,0.3),0.0)) # "inverse-ish"
#        netLyr[l].mn.append(multiplet_neuron(netLyr[l].wtgrp[i],1,1,np.random.laplace(0.5,0.3),0.0))  # "dot product"
#        netLyr[l].mn.append(multiplet_neuron(netLyr[l].wtgrp[i],1,2,np.random.laplace(0.5,0.3),0.0)) # "multiplication-ish"
#        netLyr[l].mn.append(multiplet_neuron(netLyr[l].wtgrp[i],5,1,np.random.laplace(0.5,0.3),0.0)) # "max-ish"
       

netLyr.append(mmlp_layer())  # Layer 2
for l in range(2,3): 
    vecSize = len(netLyr[l-1].mn)  # number of neurons in previous layer
    #  define layer weight groups
    # initialize the membership selection weights w_i to 1.0 
    netLyr[l].wtgrp.append(multiplet_weight_vec(np.ones((vecSize,))))
    # now attach neurons associated with these w groups
    i = 0
    netLyr[l].mn.append(multiplet_neuron(netLyr[l].wtgrp[i],-3,1,(-1*np.random.laplace(0.5,0.3)),1.0))  # b 1 
    netLyr[l].mn.append(multiplet_neuron(netLyr[l].wtgrp[i],-3,1,np.random.laplace(0.5,0.3),0.0))  # "dot product"


"""
    MAIN LOOP over EPOCHS


████████╗██████╗ ███╗   ██╗
╚══██╔══╝██╔══██╗████╗  ██║
   ██║   ██████╔╝██╔██╗ ██║
   ██║   ██╔══██╗██║╚██╗██║
   ██║   ██║  ██║██║ ╚████║
   ╚═╝   ╚═╝  ╚═╝╚═╝  ╚═══╝
                           
"""




for k in range(3):
    # epochNetworkInit()
    for ii in range(len(xAryIn)):

        netLyr[0].hardsetVecOut(xAryIn[ii]) # this is the same as above loop
        netLyr[0].hardsetCategoryTargets(yTargets[ii]) # this is the same as above loop
        print('target is=',yTargets[ii])
        forwardPassInstance(xAryIn[ii],yTargets[ii])
        backpropInstance(xAryIn[ii],yTargets[ii])
    # epochEndStats()


"""

 END, TRAINING

"""

# final results
print('+++++++++++++++++++++++++ post training +++++')
for ii in range(len(xAryIn)):

    netLyr[0].hardsetVecOut(xAryIn[ii]) 
    netLyr[0].hardsetCategoryTargets(yTargets[ii]) 
    #print('target is=',yTargets[ii])
    forwardPassInstance(xAryIn[ii],yTargets[ii])
    print('output from input vector ',xAryIn[ii],' is ')
    print(netLyr[l].vecOutI)

"""
 In only 3 iterations:  
+++++++++++++++++++++++++ post training +++++
------- layer ------  1  --------
neuron 0 aout=1.12 on p=5 b=0.00 m=1.13
      w= [1. 1.]
neuron 1 aout=0.01 on p=-3 b=1.00 m=-1.00
      w= [1. 1.]
------- layer ------  2  --------
neuron 0 aout=0.99 on p=-3 b=1.00 m=-0.95
      w= [1. 1.]
neuron 1 aout=0.01 on p=-3 b=0.00 m=0.94
      w= [1. 1.]
output from input vector  [0.99, 0.99]  is
[0.994 0.006]
------- layer ------  1  --------
neuron 0 aout=0.01 on p=5 b=0.00 m=1.13
      w= [1. 1.]
neuron 1 aout=0.99 on p=-3 b=1.00 m=-1.00
      w= [1. 1.]
------- layer ------  2  --------
neuron 0 aout=0.99 on p=-3 b=1.00 m=-0.95
      w= [1. 1.]
neuron 1 aout=0.01 on p=-3 b=0.00 m=0.94
      w= [1. 1.]
output from input vector  [0.01, 0.01]  is
[0.989 0.011]
------- layer ------  1  --------
neuron 0 aout=1.12 on p=5 b=0.00 m=1.13
      w= [1. 1.]
neuron 1 aout=0.99 on p=-3 b=1.00 m=-1.00
      w= [1. 1.]
------- layer ------  2  --------
neuron 0 aout=0.02 on p=-3 b=1.00 m=-0.95
      w= [1. 1.]
neuron 1 aout=0.97 on p=-3 b=0.00 m=0.94
      w= [1. 1.]
output from input vector  [0.99, 0.01]  is
[0.018 0.974]
------- layer ------  1  --------
neuron 0 aout=1.12 on p=5 b=0.00 m=1.13
      w= [1. 1.]
neuron 1 aout=0.99 on p=-3 b=1.00 m=-1.00
      w= [1. 1.]
------- layer ------  2  --------
neuron 0 aout=0.02 on p=-3 b=1.00 m=-0.95
      w= [1. 1.]
neuron 1 aout=0.97 on p=-3 b=0.00 m=0.94
      w= [1. 1.]
output from input vector  [0.01, 0.99]  is
[0.018 0.974]


"""   
