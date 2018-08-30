from numpy import *
import pickle, gzip
from matplotlib.pyplot import *
import scipy.linalg as linalg
import scipy
import itertools
from scipy import signal
import matplotlib as mpl
from matplotlib import pyplot as plt
import random
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
import matplotlib.cm as cm
import math
#import networkx as nx
#from networkx.generators.classic import empty_graph, path_graph, complete_graph
from scipy.sparse import coo_matrix
import math

with gzip.open('mnist.pkl.gz', 'rb') as f:
     train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
#train_set, valid_set, test_set = cPickle.load(f)  #valid set is used for similar test
f.close()#data= loadtxt('logistic.txt')

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

g = open('blurandrotate.txt', 'w')
insize=28
start=0

dt = int(len(train_set[0][1])/insize)

siz=1000
hi=[]
g.write('insize, dt '+str(insize)+str(dt) + '\n')


#generate a reservoir

lam=1.0

random.seed(42)
adj_res=(scipy.sparse.rand(siz, siz, density=0.1, format='coo', random_state=100).A-0.5)*2
adj_res[adj_res==-1]=0


adj_inp=(scipy.sparse.rand(siz, insize+1, density=0.1, format='coo', random_state=100).A-0.5)*2
adj_inp[adj_inp==-1]=0

#spectral radius (largest eigenvalue of adj matrix) for res system ~ 0.9

rho= math.sqrt(max(abs(linalg.eig(adj_res)[0])))
print ('rho ',rho)
adj_res *= 0.5/ rho

rho1= max(abs(linalg.eig(dot(adj_inp.T,adj_inp))[0]))
print ('rho1',rho1)


coo=6
kernel = np.ones((coo, coo), dtype="float") * (1.0 / (coo * coo))
noiz=np.random.rand(insize,dt)*0.2


random.seed(24)
boo=70
nono=5

l=np.where(train_set[1]<=5)[0]
for cho in range(nono):
     for i in range(boo):
         a1=random.randint(0,len(l)-1)
         a=l[a1]

         data1 = train_set[0][a]
         data1=data1.reshape(dt,insize)
         val1=train_set[1][a]
         
         if cho==0:
             f1=np.where(train_set[1]==val1)[0]
             flen1=len(f1)
             b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
             data2= train_set[0][b] #very similar
             data2=data2.reshape(dt,insize)+noiz
             val2=train_set[1][b]
             print ('same')

         if cho==1:
             f1=np.where(train_set[1]==val1)[0]
             flen1=len(f1)
             b=f1[random.randint(0,flen1-1)] #random.randint(0,len(train_set[0]-1))
             data2= train_set[0][b] #different
             data2=data2.reshape(dt,insize)
             data2=fliplr(data2.T)

             val2=train_set[1][b]

             print ('rotated')
         if cho==2:
             f1=np.where(train_set[1]==val1)[0]
             flen1=len(f1)
             b=f1[random.randint(0,flen1-1)] #random.randint(0,len(train_set[0]-1))
             data2= train_set[0][b] #blur
             data2=data2.reshape(dt,insize)

             data2=scipy.signal.convolve2d(data2, kernel, boundary='wrap', mode='same')/kernel.sum()


             val2=train_set[1][b]
             print ('blurred')

         if cho==3:
             f1=np.where(train_set[1]==val1)[0]
             flen1=len(f1)
             b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
             data2= train_set[0][b] #similar
             val2=train_set[1][b]
             data2=data2.reshape(dt,insize)
             data2=np.repeat(data2, 2, axis=1)
             data2=np.repeat(data2, 2, axis=0)
             q=shape(data2)[0]
             data2=data2[int(q/4):int(3*q/4),int(q/4):int(3*q/4)]

         if cho==4:
             f1=np.where(train_set[1]!=val1)[0]
             flen1=len(f1)
             b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
             data2= train_set[0][b] #different
             data2=data2.reshape(dt,insize)
             val2=train_set[1][b]

         x1=zeros((siz,1))
         x2=zeros((siz,1))

         #run the reservoir

         print ('running reservoir')
         err=1e-8



         for t in range(dt+start):
             a1=zeros((dt,1))
             a1[:,0]=data1[:,t]
             x1=lam*tanh(dot(adj_inp,vstack((1,a1))) + dot(adj_res,x1)) #reservoir update step
             a2=zeros((dt,1))
             a2[:,0]=data2[:,t]
             x2=lam*tanh(dot(adj_inp,vstack((1,a2))) + dot(adj_res,x2)) #reservoir update step


             if t==start:
                 Xr=abs(x1-x2)
             else:
                 Xr=hstack((Xr,abs(x1-x2)))
         y=zeros((nono,dt))
         y[cho,:]=1 #blurred


         if i==0 and cho==0:
             X=Xr
             Y=y
         else:
             X=hstack((X,Xr))
             Y=hstack((Y,y))



clf = KernelRidge(alpha=1)
clf.fit(X.T, Y.T) 
Wpred = dot(dot(Y,X.T), linalg.inv(dot(X,X.T) + err*eye(dot(X,X.T).shape[0]) ) )
print ('yay')


#prediction state

correct=0
wrong=0
lout=np.where(test_set[1]>5)[0]
vs=[]
rot=[]
blur=[]
scale=[]
diff=[]
l=100
for i in range(l):

        posout=random.randint(0,len(lout)-1)
        posout1=lout[posout]

        testdat1= train_set[0][posout1].reshape(dt,insize)
        testval1=train_set[1][posout1]
        f1=np.where(train_set[1]==val1)[0]
        flen1=len(f1)
        b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
        testdat2= train_set[0][b].reshape(dt,insize) +noiz#very similar

        #testdat2=scipy.signal.convolve2d(testdat1, kernel, boundary='wrap', mode='same')/kernel.sum()
        
        testdat2=np.repeat(testdat2, 2, axis=1)
        testdat2=np.repeat(testdat2, 2, axis=0)
        q=shape(testdat2)[0]
        testdat2=testdat2[int(q/4):int(3*q/4),int(q/4):int(3*q/4)]
        
        #testdat2=fliplr(testdat2.T)

        choout=random.randint(0,1)


        xout1=zeros((siz,1))
        xout2=zeros((siz,1))
        for t in range(dt+start):
                aout1=zeros((dt,1))
                aout2=zeros((dt,1))
                aout1[:,0]=testdat1[:,t]
                aout2[:,0]=testdat2[:,t]   
                xout1=lam*tanh(dot(adj_inp,vstack((1,aout1))) + dot(adj_res,xout1)) #reservoir update step
                xout2=lam*tanh(dot(adj_inp,vstack((1,aout2))) + dot(adj_res,xout2)) #reservoir update step
                if t==start:
                        Xout1=abs(xout1-xout2)
                else:
                        Xout1=hstack((Xout1,abs(xout1-xout2)))



        Predlabel1=clf.predict(Xout1.T) #dot(Xout1.T,Wpred.T)


        sel=Predlabel1.mean(axis=0)
        vs.append(sel[0])
        rot.append(sel[1])
        blur.append(sel[2])
        scale.append(sel[3])
        diff.append(sel[4])
        sel=sel/sum(sel)
        most=sel.argsort()[-2:][::-1] #2 maximum values

        print('test value largest value and distribution', testval1,most, sel)

        mostest=np.where(sel==most)[0]

        if 0 in most and 3 in most:
             correct+=1
        else:
             wrong+=1
             
print (correct)
print('percentage correct rate for ',i+1,' tries', correct/float(i+1))

low=min(min(vs), min(blur), min(scale),min(rot),min(diff))
if low<0:
     low=abs(low)
     vs=vs+low
     blur=blur+low
     scale=scale+low
     rot=rot+low
     diff=diff+low

for i in range(l):
     s=vs[i]+blur[i]+scale[i]+rot[i]+diff[i]
     vs[i]=vs[i]/s
     scale[i]=scale[i]/s
     rot[i]=rot[i]/s
     diff[i]=diff[i]/s
     blur[i]=blur[i]/s

plt.plot(np.linspace(0,l,l),vs,color='g')#, label=str(i))
plt.plot(np.linspace(0,l,l),rot,color='r')#,color=colors[i], label=str(i))
plt.plot(np.linspace(0,l,l),blur,color='b')#,color=colors[i], label=str(i))
plt.plot(np.linspace(0,l,l),scale,color='c')#,color=colors[i], label=str(i))
plt.plot(np.linspace(0,l,l),diff,color='k')#,color=colors[i], label=str(i))
plt.xlabel('$Iteration$', fontsize = 20)
plt.legend(['Noise' ,'Rotation', 'Blur', 'Scaling', 'Different'], loc='upper right',fontsize=15)
plt.ylabel('$Label \quad Probability$', fontsize = 20)
plt.ylim([0,1])
plt.xticks(fontsize = 15)
plt.yticks(fontsize = 15)

plt.tight_layout()
#plt.title('Node '+ str(i))
plt.savefig('dual_scalennoiz'+"{0:.0f}".format(correct)+'.jpg')
plt.close()

