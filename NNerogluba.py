import pandas as pd
import tensorflow
import numpy as np
from numpy import exp 
from numpy import transpose as tp
import sklearn
from sklearn import preprocessing


class Dense:
    def __init__(self,neurons,i):
        self.neurons = neurons
        self.layernumber = i
        
    def MatrixInitialize(self,neurlist,inpshape): 
        if self.layernumber == 0:
            self.mat = abs(np.random.randn(self.neurons,inpshape)*np.sqrt(1/4))
        else:
            self.mat = abs(np.random.randn(self.neurons,neur[self.layernumber-1])*np.sqrt(1/4)+1)
            
    def BiasInitialize(self,neurlist):
        if self.layernumber != 0  :
            self.bias = np.random.randn(neur[self.layernumber],1)
        else:
            self.bias = np.zeros((neur[self.layernumber],1))
            
    def Matrix(self,ins):
        return self.mat@ins
    
    def ReLu(self,ins):
        return ins * (ins > 0)
        #return 1/(1+np.exp(-ins))
    
    def DerReLu(self,ins):
        return np.diag(np.ndarray.flatten((ins>=0)*1))
        #return np.diag(np.ndarray.flatten(np.divide(exp(-ins),(1+exp(-ins))**2)))
    
    def DerReLuLayer(self,inp):
        return self.DerReLu(self.Matrix(inp)+self.bias)

            
    def GradientProp(self,inp):
        return self.DerReLuLayer(inp)@self.mat
    
    def LayerOut(self,ins):
        ins = self.Matrix(ins)+self.bias
        return self.ReLu(ins)

        
def Setup(neur,inpshape):    
    classlist = []
    for i in range(len(neur)):
        classlist.append(Dense(neur[i],i))
        classlist[i].MatrixInitialize(neur,inpshape)
        classlist[i].BiasInitialize(neur)
    return classlist
def NetworkPass(classlist,inp1): #One pass through the network with 1 instance
    n_samples = inp.shape[0]
    for i in range(len(classlist)):
        pass_out = classlist[i].LayerOut(inp1)
        inp1 = pass_out
    return pass_out

def NetworkError(inp,out):
    n_samples = inp.shape[0]
    e_array = []
    for k in range(n_samples):
        inp1 = inp[k]
        out1 = out[k,:]
        out_e = NetworkPass(classlist,inp1)   
        e_array.append(float(out1 - out_e))
    e_array = np.asarray(e_array)
    mse = np.dot(e_array,e_array)
    return e_array,mse

def LayerInputs(inp):
    inplist=[inp]
    for i in range(len(classlist)):
        if i != max(range(len(classlist))):
            inp = classlist[i].LayerOut(inp)
            inplist.append(inp)
    return inplist
def GradientChain(inp):
    
    dProp = []
    dRelu = []
    chain = []
    for i in range(len(classlist)):
        dPr = classlist[i].GradientProp(inp[i])
        dRe = classlist[i].DerReLuLayer(inp[i])
        dProp.append(dPr)
        dRelu.append(dRe)
    for i in range(len(classlist)-1):
        mult = np.identity(dProp[i+1].shape[1])
        for j in range(i+1,len(classlist)):
            mult = dProp[j]@mult
        mult = mult@dRelu[i]
        chain.append(mult)
    chain.append(dRelu[-1])
    return chain


def NetworkGradient(classlist,inp,out):
    storage = []
    dif,mse = NetworkError(inp,out)
    for k in range(inp.shape[0]):
        inplist= LayerInputs(inp[k])
        gchain = GradientChain(inplist)
        for i in range(len(classlist)):
            layer_inp = inplist[i] 
            [row_no,column_no] = classlist[i].mat.shape
            dmat = np.zeros((row_no,column_no+1))
            for j in range(column_no+1):
                if j != column_no:
                    id_input = (np.identity(row_no)*float(layer_inp[j]))
                else:
                    id_input = (np.identity(row_no))
                dodj = gchain[i]@id_input                      
                dedj = (-2*dif[k]*dodj)
                dmat[:,j] = dedj
            if k == 0:
                storage.append(dmat) 
            else:
                storage[i] += dmat
    return storage
            
u =np.asarray(pd.read_csv('in.dat',sep = ',',header = None))
scaler = preprocessing.MinMaxScaler()
u = scaler.fit_transform(u)
inp = np.reshape(u,(u.shape[0],u.shape[1],1))

y = np.reshape(np.asarray(pd.read_csv('out.dat',sep=',',header = None)),(-1,1))
scaler2 = preprocessing.MinMaxScaler()
out = scaler2.fit_transform(y)

neur = [4,1]
input_dim = inp.shape[1]
classlist = Setup(neur,input_dim)

mse = 1
while mse > 1e-3:    
        st = []
        st = NetworkGradient(classlist,inp,out)
        for i in range(len(classlist)):
            classlist[i].mat = classlist[i].mat - 0.0001*st[i][:,0:-1]
            if i != 0:
                classlist[i].bias = classlist[i].bias - 0.0001*np.reshape(st[i][0,-1],(-1,1))
        su,ms = NetworkError(inp,out)
        print('MSE Loss: %.6f'%(ms))
        
        
        
        