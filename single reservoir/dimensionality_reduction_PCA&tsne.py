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



     adj_inp=(scipy.sparse.rand(int(siz/2), insize+1, density=0.10, format='coo', random_state=100).A-0.5)*2
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
                       d='noise'

                   if cho==4:
                       f1=np.where(train_set[1]!=val1)[0]
                       flen1=len(f1)
                       b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
                       data2= train_set[0][b] #different
                       data2=data2.reshape(dt,insize)
                       val2=train_set[1][b]
                       d='different'

                   if cho==1:
                       f1=np.where(train_set[1]==val1)[0]
                       flen1=len(f1)
                       b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
                       data2= train_set[0][b] #similar
                       val2=train_set[1][b]
                       data2=data2.reshape(dt,insize)
                       data2=fliplr(data2.T)
                       d='rotate'

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
                       d='scale'
            
                   if cho==2:
                       f1=np.where(train_set[1]==val1)[0]
                       flen1=len(f1)
                       b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
                       data2= train_set[0][b] #similar
                       val2=train_set[1][b]
                       data2=data2.reshape(dt,insize)

                       data2= scipy.signal.convolve2d(data2, kernel, boundary='wrap', mode='same')/kernel.sum()
                       d='blur'
                       
                   if cho==5 and i<(boo/4):  #rotnblur
                       f1=np.where(train_set[1]==val1)[0]
                       flen1=len(f1)
                       b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
                       data2= train_set[0][b]
                       val2=train_set[1][b]
                       data2=data2.reshape(dt,insize)

                       data2= scipy.signal.convolve2d(data2, kernel, boundary='wrap', mode='same')/kernel.sum()
                       data2=fliplr(data2.T)
                       d='rotate+blur'

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
                       d='noise+scale'
                    
                   

                   x1=zeros((siz,1))
                   x2=zeros((siz,1))
                   #x3=zeros((siz,1))



                   #run the reservoir

                   #print ('running reservoir')
                   err=1e-8



                   for t in range(dt+start):
                       a1=zeros((dt,1))
                       a1[:,0]=data1[:,t]
                       a2=zeros((dt,1))
                       a2[:,0]=data2[:,t]
                       in1=dot(adj_inp,vstack((1,a1)))
                       in2=dot(adj_inp,vstack((1,a2)))
                       intot=np.concatenate((in1,in2))
                       x1=lam*tanh(intot + dot(adj_res,x1)) #reservoir update step
                       '''a2=zeros((dt,1))
                       a2[:,0]=data2[:,t]
                       x2=lam*tanh(dot(adj_inp,vstack((1,a2))) + dot(adj_res,x2)+store) #reservoir update step'''
                       '''a3=zeros((dt,1))
                       a3[:,0]=data3[:,t]
                       x3=lam*tanh(dot(adj_inp,vstack((1,a3))) + dot(adj_res,x3)) #reservoir update step'''
                       #print ('hi' ,(x1-x2))


                       if t==start:
                           Xr=(x1)
                       else:
                           Xr=hstack((Xr,(x1)))
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
                               out1=dot(adj_inp,vstack((1,aout1)))
                               out2=dot(adj_inp,vstack((1,aout2)))
                               outtot=np.concatenate((out1,out2))
                                        
                               xout1=lam*tanh(outtot + dot(adj_res,xout1)) #reservoir update step
                               if t==start:
                                       Xout1=(xout1)
                               else:
                                       Xout1=hstack((Xout1,(xout1)))


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

np.savetxt('single_pca_tsne_data.txt',tsne_pca_results)

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

np.savetxt('single_pca_tsne_data.txt',tsne_pca_results)

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
plt.savefig('pca+tsne_single.jpeg',bbox_inches='tight')




'''a=corr2(Hi0,HI0)
     b=corr2(Hi0,HI1)
     c=corr2(Hi0,HI2)
     d=corr2(Hi1,HI0)
     e=corr2(Hi1,HI1)
     f=corr2(Hi1,HI2)
     j=corr2(Hi1,HI0)
     h=corr2(Hi1,HI1)
     i=corr2(Hi1,HI2)

     print(corr2(Hi0,Hi0_),corr2(Hi0,Hi1_), corr2(Hi0,Hi2_), corr2(Hi1,Hi0_), corr2(Hi1,Hi1_), corr2(Hi1,Hi2_),corr2(Hi2,Hi0_),corr2(Hi2,Hi1_), corr2(Hi2,Hi2_),'old')
     #print(corr2(Hi00,Hi11),corr2(Hi00,Hi22), corr2(Hi11,Hi22), 'new_smaller')

     g.write('same digit \n'+str(a)+ '\t' + str(b) + '\t' + str(c) + '\n' + str(d)+ '\t' +  str(e) + '\t' + str(f) + '\n' + str(j)+ '\t' + str(h) + '\t' + str(i) + '\n\n' )

     a=corr2(HI0,Hi0_)
     b=corr2(HI0,Hi1_)
     c=corr2(HI0,Hi2_)
     d=corr2(HI1,Hi0_)
     e=corr2(HI1,Hi1_)
     f=corr2(HI1,Hi2_)
     j=corr2(HI1,Hi0_)
     h=corr2(HI1,Hi1_)
     i=corr2(HI1,Hi2_)

     g.write('diffdigits \n'+str(a)+ '\t' + str(b) + '\t' + str(c) + '\n' + str(d)+ '\t' +  str(e) + '\t' + str(f) + '\n' + str(j)+ '\t' + str(h) + '\t' + str(i) + '\n\n' )
                     
     g.write(' number tried ' + str(i+1) + '\t' + ' number correct ' + str(correct)+ 'error rate =' + str(1-float(correct/i+1)))


     del Hi0, Hi0_, HI0, HI0_
     del Hi1, Hi1_, HI1, HI1_
     del Hi2, Hi2_, HI2, HI2_'''





'''
    
    hi[3*i][1]=X2
    hi[3*i][2]=X2-X1
    hi[3*i][3]=val1
    hi[3*i][4]=val2
    hi[3*i][5]=[1,0,0]


np.savetxt('statenlabel.txt',hi)

    X1=reshape(X1, (size(X1),1), order='F')
    X2=reshape(X2, (size(X2),1), order='F')


    print('hi', X1)

    #regression for training
    label=zeros((10,1)) #very similar, similar, dissimilar. similar is for rotated/blurred images
    label[val,0]=1

    print(label, 'label')


    K=dot(label,X1.T) 
    print (K.shape, X1.shape)
    M=linalg.pinv(X1[0:1000,0])
    #adj_pred=dot( K,dot(label,X1.T), linalg.inv( dot(X1,X1.T) + err*eye(size(X1)))) # 1+insize+siz) ) )

    plot( X1[0:20,0:200].T )
    title('Activation X matrix $\mathbf{x}(n)$')
    legend()
    show()


#train it


#ss=data[None,start:start+dt]
ss=data[start+1:start+dt+1].T
#adj_pred=dot( dot(ss,X.T), linalg.inv( dot(X,X.T) + err*eye(1+insize+siz) ) )
adj_pred= dot(ss,linalg.pinv(X)) #let's find the adj_mat that causes the output to be the same as input.


#training was from t=1000 to t=2000. prediction starts from t=2000-3000
Y=zeros((1,testsiz))
a1=data[start+dt]
for t in range(testsiz):
    x=(1-lam)*x + lam*tanh(dot(adj_inp,vstack((1,a1))) + dot(adj_res,x))
    pred= dot(adj_pred, vstack((1,a1,x)))
    Y[:,t]=pred
    a1=pred #updating the new generated value at time t
print adj_pred.shape, 'hi'   
imshow(adj_pred[0:25, None])
show()
#what's the error?
mse=sum(square(data[start+dt+1:start+dt+testsiz+1]-Y[0,:testsiz]))/testsiz
print 'meansquare error: ', mse

figure(1).clear()
plot(data[start+dt:start+dt+testsiz], color='b', label = 'real_data')
plot(Y[0,:testsiz], color='r', label='simulated')
title('Reservoir computing: Data from region %.1f s - %.1f s' %(start+dt, start+dt+testsiz))
legend()
show()
              
'''  
    
    
