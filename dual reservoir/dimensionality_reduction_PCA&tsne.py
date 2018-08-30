from numpy import *
import pickle, gzip
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.pyplot import *
import scipy.linalg as linalg
import scipy
import itertools
from sklearn.decomposition import PCA
from scipy import signal
import random
import json
from copy import deepcopy
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
import matplotlib.cm as cm
from matplotlib import ticker
import math
#import networkx as nx
#from networkx.generators.classic import empty_graph, path_graph, complete_graph
from scipy.sparse import coo_matrix
import math
mpl.rcParams['xtick.labelsize'] =16
mpl.rcParams['ytick.labelsize'] =16

#doesn't work in any limit of connection between the reservoirs!

with gzip.open('mnist.pkl.gz', 'rb') as f:
     train_set, valid_set, test_set = pickle.load(f,encoding='latin1')
#train_set, valid_set, test_set = cPickle.load(f)  #valid set is used for similar test
f.close()#data= loadtxt('logistic.txt')

def corr2(a,b):
    a = a - mean2(a)
    b = b - mean2(b)

    r = (a*b).sum() / math.sqrt((a*a).sum() * (b*b).sum());
    return r

def plot_mnist_digit(image):
    """ Plot a single MNIST image."""

    fig = figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(image, cmap = matplotlib.cm.binary)
    xticks(np.array([]))
    yticks(np.array([]))
    show()

def mean2(x):
    y = np.sum(x) / np.size(x);
    return y

insize=28
start=0

dt = int(len(train_set[0][1])/insize)
random.seed(42)

X_hi=[]
y_hi=[]

res_size_single={}
sizz=linspace(1000,1000,2)
for r in range(len(sizz)-1):

     siz=int(sizz[r+1])#1000
     print (siz)
     hi=[]


     #generate a reservoir

     lam=1.0

     random.seed(42)
     adj_res=(scipy.sparse.rand(siz, siz, density=0.10, format='coo', random_state=100).A-0.5)*2
     adj_res[adj_res==-1]=0



     adj_inp=(scipy.sparse.rand(siz, insize+1, density=0.10, format='coo', random_state=100).A-0.5)*2
     adj_inp[adj_inp==-1]=0
     print (shape(adj_inp))

     #spectral radius (largest eigenvalue of adj matrix) for res system ~ 0.9

     rho= math.sqrt(max(abs(linalg.eig(adj_res)[0])))
     print ('rho ',rho)
     for soo in range(1):
          cv=0.5#0.1+ 0.1*soo
          adj_res *= cv/ rho

          rho1= max(abs(linalg.eig(dot(adj_inp.T,adj_inp))[0]))
          print ('rho1',rho1)
          coo=6
          kernel = np.ones((coo, coo), dtype="float") * (1.0 / (coo * coo))
          '''kernel=np.array([[0, 0, 1, 0, 0],
                             [0, 1, 1, 1, 0],
                             [1, 1, 1, 1, 1],
                             [0, 1, 1, 1, 0],
                             [0, 0, 1, 0, 0]])/13'''
          l=np.where(train_set[1]<5)[0]
         
          trans=[]

          labels=5
          boo=500
          trans=[]
          for cho in range(labels+1):
               print (cho)
               for i in range(boo):
                   a=random.randint(0,len(train_set[0]-1))
                   a1=random.randint(0,len(l)-1)
                   a=l[a1]
                   data1 = train_set[0][a]
                   data1=data1.reshape(dt,insize)
                   val1=train_set[1][a]
                   noiz=np.random.rand(insize,dt)*0.2 #scipy.sparse.rand(insize,dt, density=0.7).A*0.4
                   #print (noiz)

                   if cho==0:
                       f1=np.where(train_set[1]==val1)[0]
                       flen1=len(f1)
                       b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
                       data2= train_set[0][b] #very similar
                       data2=data2.reshape(dt,insize)+noiz
                       val2=train_set[1][b]

                   if cho==4:
                       f1=np.where(train_set[1]!=val1)[0]
                       flen1=len(f1)
                       b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
                       data2= train_set[0][b] #different
                       data2=data2.reshape(dt,insize)
                       val2=train_set[1][b]

                   if cho==1:
                       f1=np.where(train_set[1]==val1)[0]
                       flen1=len(f1)
                       b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
                       data2= train_set[0][b] #similar
                       val2=train_set[1][b]
                       data2=data2.reshape(dt,insize)
                       data2=fliplr(data2.T)

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
            
                   if cho==2:
                       f1=np.where(train_set[1]==val1)[0]
                       flen1=len(f1)
                       b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
                       data2= train_set[0][b] #similar
                       val2=train_set[1][b]
                       data2=data2.reshape(dt,insize)

                       data2= scipy.signal.convolve2d(data2, kernel, boundary='wrap', mode='same')/kernel.sum()
                       
                   if cho==5 and i<(boo/4):  #rotnblur
                       f1=np.where(train_set[1]==val1)[0]
                       flen1=len(f1)
                       b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
                       data2= train_set[0][b]
                       val2=train_set[1][b]
                       data2=data2.reshape(dt,insize)

                       data2= scipy.signal.convolve2d(data2, kernel, boundary='wrap', mode='same')/kernel.sum()
                       data2=fliplr(data2.T)

                   if cho==6 and i<(boo/4): #scale+noiz
                       f1=np.where(train_set[1]==val1)[0]
                       flen1=len(f1)
                       b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
                       data2= train_set[0][b] 
                       val2=train_set[1][b]
                       data2=data2.reshape(dt,insize)
                       data2=np.repeat(data2, 2, axis=1)
                       data2=np.repeat(data2, 2, axis=0)
                       q=shape(data2)[0]
                       data2=data2[int(q/4):int(3*q/4),int(q/4):int(3*q/4)]
                       data2=data2+noiz
                    
                   

                   x1=zeros((siz,1))
                   x2=zeros((siz,1))
                   #x3=zeros((siz,1))



                   #run the reservoir

                   #print ('running reservoir')
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
                   y=zeros((labels+2,dt))
                   y[cho,:]=1 




                   if i==0 and cho==0:
                       X=Xr
                       Y=y
                   else:
                       X=hstack((X,Xr))
                       Y=hstack((Y,y))
                   X_hi.append(np.ravel(Xr))
                   y_hi.append(cho)

                  

          break

          

          clf = KernelRidge(alpha=1)
          clf.fit(X.T, Y.T) 
          Wpred = dot(dot(Y,X.T), linalg.inv(dot(X,X.T) + err*eye(dot(X,X.T).shape[0]) ) )
          print ('yay')
          nodes=50





          #prediction state


          lout=np.where(test_set[1]>5)[0]
          total=0
          for jiggs in range(labels):
               correct=0
               wrong=0
               for i in range(100):
                       posout=random.randint(0,len(lout)-1)
                       posout1=lout[posout] #random.randint(0,len(train_set[1])-1)
                       posout2=random.randint(0,len(train_set[1])-1)


                       testdat1= train_set[0][posout1].reshape(dt,insize)
                       testval1=train_set[1][posout1]
                       choout=jiggs#random.randint(0,3)
                       if choout==0:
                            gree=np.where(test_set[1]==testval1)[0]
                            blu=random.randint(0,len(gree)-1)
                            goo=gree[blu]
                            testdat2= train_set[0][goo].reshape(dt,insize)+noiz
                            testval2=train_set[1][goo]
                       if choout==1:
                            gree=np.where(test_set[1]==testval1)[0]
                            blu=random.randint(0,len(gree)-1)
                            goo=gree[blu]
                            testdat2= fliplr(train_set[0][goo].reshape(dt,insize).T)
                            testval2=train_set[1][goo]
                       if choout==3:
                            gree=np.where(test_set[1]==testval1)[0]
                            blu=random.randint(0,len(gree)-1)
                            goo=gree[blu]
                            testdat2= train_set[0][goo].reshape(dt,insize)
                            testdat2=np.repeat(testdat2, 2, axis=1)
                            testdat2=np.repeat(testdat2, 2, axis=0)
                            testdat2=testdat2[int(q/4):int(3*q/4),int(q/4):int(3*q/4)]
                            testval2=train_set[1][goo]
                       if choout==2:
                            gree=np.where(test_set[1]==testval1)[0]
                            blu=random.randint(0,len(gree)-1)
                            goo=gree[blu]
                            testdat2= scipy.signal.convolve2d(train_set[0][goo].reshape(dt,insize), kernel, boundary='wrap', mode='same')/kernel.sum()
                            testval2=train_set[1][goo]
                       '''if choout==2:
                            gree=np.where(test_set[1]==testval1)[0]
                            blu=random.randint(0,len(gree)-1)
                            goo=gree[blu]
                            testdat2= train_set[0][goo].reshape(dt,insize)
                            testval2=train_set[1][goo]
                            dataa2=testdat2
                            for qq in range(shape(data2)[0]-8):
                                 for ww in range(shape(data2)[0]-8):
                                      testdat2[4+qq,4+ww]=dataa2[4+qq+int(4*math.sin(2*math.pi*(4+ww)/128)),4+ww]'''
                       if choout==4:
                            gree=np.where(test_set[1]!=testval1)[0]
                            blu=random.randint(0,len(gree)-1)
                            goo=gree[blu]
                            testdat2= train_set[0][goo].reshape(dt,insize)
                            testval2=train_set[1][goo]




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
                       Predlabel1=dot(Xout1.T,Wpred.T)

                       sel=Predlabel1.mean(axis=0)
                       sel=sel/sum(sel)
                       most=max(sel)

                       mostest=np.where(sel==most)[0]
                       #print('expected and predicted', mostest, choout)

                       if mostest==choout:
                               correct+=1
                               total+=1
                       if mostest!=choout:
                               wrong+=1

               print('success rate for size ',siz,'/t',jiggs, correct/(i+1))
          total= total/((jiggs+1)*(i+1))
          res_size_single[siz]=total
          print ('total:',total)





X=np.array(X_hi)
y=np.array(y_hi)


import pandas as pd


feat_cols = [ 'pixel'+str(i) for i in range(X.shape[1]) ]

df = pd.DataFrame(X,columns=feat_cols)
df['label'] = y 
df['label'] = df['label'].apply(lambda i: str(i))

#X, y = None, None

rndperm = np.random.permutation(df.shape[0])

from sklearn.decomposition import PCA


'''pca = PCA(n_components=3)
pca_result = pca.fit_transform(df[feat_cols].values)

df['pca-one'] = pca_result[:,0]
df['pca-two'] = pca_result[:,1] 
df['pca-three'] = pca_result[:,2]'''

from ggplot import *

n_sne = 7000

'''chart = ggplot( df.loc[rndperm[:n_sne],:], aes(x='pca-one', y='pca-two', color='label') ) \
        + geom_point(size=75,alpha=0.8) \
        + ggtitle("1st and 2nd Principal Components \n colored by Transformation")
p=chart#+t
p.save('PCA_single.png')'''



from sklearn.manifold import TSNE




'''tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300,method='exact')
tsne_results = tsne.fit_transform(df.loc[rndperm[:n_sne],feat_cols].values)


df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne['x-tsne'] = tsne_results[:,0]
df_tsne['y-tsne'] = tsne_results[:,1]

chart = ggplot( df_tsne, aes(x='x-tsne', y='y-tsne', color='label') ) \
        + geom_point(size=70,alpha=0.4) \
        + ggtitle("tSNE Dimensions colored by Transformation")'''
'''t = theme_gray()
t._rcParams['font.size'] = 30
t._rcParams['xtick.labelsize'] = 25
t._rcParams['ytick.labelsize'] = 25'''
#p=chart#+t
#p.save('tsne_single.png')


pca_50 = PCA(n_components=100)
pca_result_50 = pca_50.fit_transform(df[feat_cols].values)
print ('Explained variation per principal component (PCA): {}'.format(np.sum(pca_50.explained_variance_ratio_)))

tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, method='exact')
tsne_pca_results = tsne.fit_transform(pca_result_50[rndperm[:n_sne]])

np.savetxt('dual_pca_tsne_data.txt',tsne_pca_results)

df_tsne = None
#df_tsne = df.loc[rndperm[:n_sne],:].copy()
df_tsne = df.loc[:,:].copy()
df_tsne['x-tsne-pca'] = tsne_pca_results[:,0]
df_tsne['y-tsne-pca'] = tsne_pca_results[:,1]

color=['g','r','b','c','k','orange','purple']

#chart = ggplot( df_tsne, aes(x='x-tsne-pca', y='y-tsne-pca', color='label') ) \
#        + geom_point(size=70,alpha=0.3) \
#        + ggtitle("tSNE Dimensions colored by Transformation (PCA)")
#p=chart

np.savetxt('dual_pca_tsne_data.txt',tsne_pca_results)

color=['g','r','b','c','k','orange','purple']

col=[]
for j in range(len(color)):
     for i in range(500):
          col.append(color[j])

colo=[col[i] for i in rndperm[:n_sne]]
indexes=np.where(np.array(colo)=='purple')[0] #don't need noise+zoom


x=tsne_pca_results[:,0]
y=tsne_pca_results[:,1]
import matplotlib.pyplot as plt
labels=['Noise','Rotate 90','Blur','Zoom','Different','Rotate+Blur','Noise+Zoom']
print (len (colo))
for index in sorted(indexes, reverse=True):
    x=np.delete(x,index)
    y=np.delete(y,index)
    colo=np.delete(colo,index)

print (len (colo))
plt.figure()

plt.scatter(x,y,color=colo,alpha=0.15,marker='o',s=40)
for i in range(len(color)-1):
    c=color[i]
    print
    plt.scatter([], [], color=color[i],
                label=labels[i])
plt.legend(scatterpoints=1, frameon=True, labelspacing=0.3, title='Transformation:',loc='upper right')
plt.ylabel('$TSNE-PCA-2$', fontsize=15)
plt.xlabel('$TSNE-PCA-1$', fontsize=15)
plt.savefig('pca+tsne_dual.jpeg',bbox_inches='tight')
