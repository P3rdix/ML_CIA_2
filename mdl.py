import pandas as pd
import numpy as np
from random import random, randint
import pickle

def sigmoid(x):
    return 1/(1 + np.exp(-1*x))

class node:
    def __init__(self,no_inputs=1, mx = 1):
        self.n = no_inputs
        self.input_1 = []
        self.weight_1 = [random() for i in range(self.n)]
        self.prev = [0]
        self.prev_weight = random()
        self.bias_1 = random()
        self.input_2 = []
        self.weight_2 = random()
        self.bias_2 = random()
        self.output = []
        self.max = mx
    
    def forward_propogate(self,inp):
        for i in inp:
            self.input_1.append(i)
            s = 0
            if self.n >1:
                for j in range(self.n):
                    s += self.input_1[-1][j]*self.weight_1[j]
            else:
                s+= self.input_1[-1]*self.weight_1[0]
            s += self.prev[-1]*self.prev_weight
            s += self.bias_1
            self.input_2.append(sigmoid(s))
            self.prev.append(self.input_2[-1])
            self.output.append(sigmoid(self.input_2[-1]*self.weight_2 + self.bias_2))
        return
    
    def back_propogate(self,real_val, learning_rate):
        error = []
        sum = 0
        c_weight_1 = [0 for i in range(len(self.weight_1))]
        c_weight_2 = 0
        c_prev_weight = 0
        c_bias_1 = 0
        c_bias_2 = 0
        carry_weight = 0
        for i in range(len(real_val)):
            error.append(real_val[i]-self.output[i])
        
        for i in range(len(self.output)-1,-1,-1):
            ex2 = self.output[i]*(1-self.output[i])*error[i]
            c_weight_2 += ex2*self.input_2[i]
            c_bias_2 += ex2
            ei2 = ex2*self.weight_2
            ex1 = (0.01*carry_weight+ei2)*self.input_2[i]*(1-self.input_2[i])
            if len(self.weight_1) > 1:
                for j in range(len(self.weight_1)):
                    c_weight_1[j] += ex1*self.input_1[i][j]
            else:
                c_weight_1[0] += ex1*self.input_1[i]
            c_bias_1 += ex1
            c_prev_weight += ex1*self.prev[i]
            carry_weight += ex1*self.prev[i]
        self.weight_2 += learning_rate*c_weight_2
        self.bias_2 += c_bias_2*learning_rate
        self.prev_weight += learning_rate*c_prev_weight
        self.bias_1 += learning_rate*c_bias_1
        for i in range(len(self.weight_1)):
            self.weight_1[i] += learning_rate*c_weight_1[i]
        return
    
    def clear(self):
        self.input_1 = []
        self.prev = [0]
        self.input_2 = []
        self.output = []

    def fit(self,inp,output,n = 1):
        for i in range(n):
            self.forward_propogate(inp)
            self.back_propogate(output, 0.00001)    
            self.clear()
    
    def predict(self,start, n = 10):
        self.input_1 = start
        for i in range(n):
            x1 = 0
            for j in range(len(self.input_1)):
                x1 += self.input_1[j]*self.weight_1[j]
            x1 += self.prev[-1]*self.prev_weight + self.bias_1
            self.input_2.append(sigmoid(x1))
            self.output.append(sigmoid(self.input_2[-1]*self.weight_2 + self.bias_2))
            self.prev.append(self.input_2[-1])
            print(self.prev[-1])
        o = self.output
        self.clear()
        for i in range(len(o)):
            o[i] = o[i]*self.max
        return o
    


f = open('Wizard_Space_Program.txt','r')
a = f.read()
rmchars = "!@#$%^&*()_+-=?/>.<,:;{[]}\|"
for i in rmchars:
    a = a.replace(i,"")
b = a.split(' ')
df = pd.DataFrame(b)
d = df[0].astype('category')
l = d.cat.codes
inp = l[:-1]
output = l.iloc[1:]
o = output.values/(len(output.values)+2)
i = inp.values/(len(inp.values)+2)
n = node(1,len(inp.values)+1)
n.fit(i,o,1)

pickle.dump(n,open('model.pkl','wb'))