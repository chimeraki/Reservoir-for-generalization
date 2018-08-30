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
g = open('final.txt', 'w')
dt = int(len(train_set[0][1])/insize)

g.write('sparsity: res = 0.9, and inp is 0.9, spectral rad =0.8, lambda=1 \n')

g.write('insize, dt '+str(insize)+str(dt) + '\n')

spectr={}
sizz=linspace(500,500,2)
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
          random.seed('None')
          trans=[]

          boo=50
          trans=[]
          for cho in range(5):
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
                       if val2==3:
                            plt.imshow(data2, cmap='Greys',  interpolation='nearest')
                            plt.savefig('very_similar_noiz.png')
                       if i==0:
                            trans.append('very similar')
                       if i<=20:
                            g.write(str(i)+'th image very similar '+str(val1) + ' value ' +'\n')
                   if cho==4:
                       f1=np.where(train_set[1]!=val1)[0]
                       flen1=len(f1)
                       b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
                       data2= train_set[0][b] #different
                       data2=data2.reshape(dt,insize)
                       val2=train_set[1][b]
                       if i<=20:
                            g.write(str(i)+'th image different '+str(val1) +'\t'+ str(val2)+ ' value ' +'\n')
                       if i==0:
                            trans.append('different')
                   if cho==1:
                       f1=np.where(train_set[1]==val1)[0]
                       flen1=len(f1)
                       b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
                       data2= train_set[0][b] #similar
                       val2=train_set[1][b]
                       data2=data2.reshape(dt,insize)
                       data2=fliplr(data2.T)
                       if i==0:
                            trans.append('rotated90')
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
                       if val2==3:
                            plt.imshow(data2, cmap='Greys',  interpolation='nearest')
                            plt.savefig('zoom.png')
                       if i==0:
                            trans.append('scale')
                   if cho==2:
                       f1=np.where(train_set[1]==val1)[0]
                       flen1=len(f1)
                       b=f1[random.randint(0,flen1-1)]#random.randint(0,len(train_set[0]-1))
                       data2= train_set[0][b] #similar
                       val2=train_set[1][b]
                       data2=data2.reshape(dt,insize)

                       data2= scipy.signal.convolve2d(data2, kernel, boundary='wrap', mode='same')/kernel.sum()
                       if val2==3:
                            plt.imshow(data2, cmap='Greys',  interpolation='nearest')
                            plt.savefig('blurred.png')
                       if i==0:
                            trans.append('blurred')



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
                       '''a3=zeros((dt,1))
                       a3[:,0]=data3[:,t]
                       x3=lam*tanh(dot(adj_inp,vstack((1,a3))) + dot(adj_res,x3)) #reservoir update step'''
                       #print ('hi' ,(x1-x2))


                       if t==start:
                           Xr=abs(x1-x2)
                       else:
                           Xr=hstack((Xr,abs(x1-x2)))
                   y=zeros((5,dt))
                   y[cho,:]=1 




                   if i==0 and cho==0:
                       X=Xr
                       Y=y
                   else:
                       X=hstack((X,Xr))
                       Y=hstack((Y,y))

                   '''if cho==0 and val1==3:
                        if i%2==0:  #splitting it into 2 for comparison
                             if 'Hi0' not in dir():
                                  Hi0=Xr
                             else:
                                  Hi0=hstack((Hi0,Xr))
                        else:

                             if 'HI0' not in dir():
                                  HI0=Xr
                             else:
                                  HI0=hstack((HI0,Xr))

                   if cho==1 and val1==3:

                        if i%2==0:
                             if 'Hi1' not in dir():
                                  Hi1=Xr
                             else:
                                  Hi1=hstack((Hi1,Xr))
                        else:

                             if 'HI1' not in dir():
                                  HI1=Xr
                             else:
                                  HI1=hstack((HI1,Xr))

                   if cho==2 and val1==3:
                        if i%2==0:
                             if 'Hi2' not in dir():
                                  Hi2=Xr
                             else:
                                  Hi2=hstack((Hi2,Xr))
                        else:

                             if 'HI2' not in dir():
                                  HI2=Xr
                             else:
                                  HI2=hstack((HI2,Xr))
                   if cho==3 and val1==3:
                        if i%2==0:
                             if 'Hi3' not in dir():
                                  Hi3=Xr
                             else:
                                  Hi3=hstack((Hi3,Xr))
                        else:

                             if 'HI3' not in dir():
                                  HI3=Xr
                             else:
                                  HI3=hstack((HI3,Xr))
                   if cho==4 and val1==3:
                        if i%2==0:
                             if 'Hi4' not in dir():
                                  Hi4=Xr
                             else:
                                  Hi4=hstack((Hi4,Xr))
                        else:

                             if 'HI4' not in dir():
                                  HI4=Xr
                             else:
                                  HI4=hstack((HI4,Xr))
                   if cho==5 and val1==3:
                        if i%2==0:
                             if 'Hi5' not in dir():
                                  Hi5=Xr
                             else:
                                  Hi5=hstack((Hi5,Xr))
                        else:

                             if 'HI5' not in dir():
                                  HI5=Xr
                             else:
                                  HI5=hstack((HI5,Xr))
                   if cho==0 and val1==4:
                        if i%2==0:
                             if 'Hi0_' not in dir():
                                  Hi0_=Xr
                             else:
                                  Hi0_=hstack((Hi0_,Xr))
                        else:

                             if 'HI0_' not in dir():
                                  HI0_=Xr
                             else:
                                  HI0_=hstack((HI0_,Xr))
                                  
                   if cho==1 and val1==4:
                        if i%2==0:
                             if 'Hi1_' not in dir():
                                  Hi1_=Xr
                             else:
                                  Hi1_=hstack((Hi1_,Xr))
                        else:

                             if 'HI1_' not in dir():
                                  HI1_=Xr
                             else:
                                  HI1_=hstack((HI1_,Xr))

                   if cho==2 and val1==4:
                        if i%2==0:
                             if 'Hi2_' not in dir():
                                  Hi2_=Xr
                             else:
                                  Hi2_=hstack((Hi2_,Xr))
                        else:

                             if 'HI2_' not in dir():
                                  HI2_=Xr
                             else:
                                  HI2_=hstack((HI2_,Xr))
                   if cho==3 and val1==4:
                        if i%2==0:
                             if 'Hi3_' not in dir():
                                  Hi3_=Xr
                             else:
                                  Hi3_=hstack((Hi3_,Xr))
                        else:

                             if 'HI3_' not in dir():
                                  HI3_=Xr
                             else:
                                  HI3_=hstack((HI3_,Xr))
                   if cho==4 and val1==4:
                        if i%2==0:
                             if 'Hi4_' not in dir():
                                  Hi4_=Xr
                             else:
                                  Hi4_=hstack((Hi4_,Xr))
                        else:

                             if 'HI4_' not in dir():
                                  HI4_=Xr
                             else:
                                  HI4_=hstack((HI4_,Xr))
                   if cho==5 and val1==4:
                        if i%2==0:
                             if 'Hi5_' not in dir():
                                  Hi5_=Xr
                             else:
                                  Hi5_=hstack((Hi5_,Xr))
                        else:

                             if 'HI5_' not in dir():
                                  HI5_=Xr
                             else:
                                  HI5_=hstack((HI5_,Xr))


          Hi=[]
          Hi0=Hi0.reshape(int(len(Hi0[0])/dt),siz,dt)
          Hi.append(Hi0)
          Hi1=Hi1.reshape(int(len(Hi1[0])/dt),siz,dt)
          Hi.append(Hi1)
          Hi2=Hi2.reshape(int(len(Hi2[0])/dt),siz,dt)
          Hi.append(Hi2)
          Hi3=Hi3.reshape(int(len(Hi3[0])/dt),siz,dt)
          Hi.append(Hi3)
          Hi4=Hi4.reshape(int(len(Hi4[0])/dt),siz,dt)
          Hi.append(Hi4)
          Hi5=Hi5.reshape(int(len(Hi5[0])/dt),siz,dt)
          Hi.append(Hi5)

          HI0=HI0.reshape(int(len(HI0[0])/dt),siz,dt)
          Hi.append(HI0)
          HI1=HI1.reshape(int(len(HI1[0])/dt),siz,dt)
          Hi.append(HI1)
          HI2=HI2.reshape(int(len(HI2[0])/dt),siz,dt)
          Hi.append(HI2)
          HI3=HI3.reshape(int(len(HI3[0])/dt),siz,dt)
          Hi.append(HI3)
          HI4=HI4.reshape(int(len(HI4[0])/dt),siz,dt)
          Hi.append(HI4)
          HI5=HI5.reshape(int(len(HI5[0])/dt),siz,dt)
          Hi.append(HI5)

          Hi0_=Hi0_.reshape(int(len(Hi0_[0])/dt),siz,dt)
          Hi.append(Hi0_)
          Hi1_=Hi1_.reshape(int(len(Hi1_[0])/dt),siz,dt)
          Hi.append(Hi1_)
          Hi2_=Hi2_.reshape(int(len(Hi2_[0])/dt),siz,dt)
          Hi.append(Hi2_)
          Hi3_=Hi3_.reshape(int(len(Hi3_[0])/dt),siz,dt)
          Hi.append(Hi3_)
          Hi4_=Hi4_.reshape(int(len(Hi4_[0])/dt),siz,dt)
          Hi.append(Hi4_)
          Hi5_=Hi5_.reshape(int(len(Hi5_[0])/dt),siz,dt)
          Hi.append(Hi5_)
          HI0_=HI0_.reshape(int(len(HI0_[0])/dt),siz,dt)
          Hi.append(HI0_)
          HI1_=HI1_.reshape(int(len(HI1_[0])/dt),siz,dt)
          Hi.append(HI1_)
          HI2_=HI2_.reshape(int(len(HI2_[0])/dt),siz,dt)
          Hi.append(HI2_)
          HI3_=HI3_.reshape(int(len(HI3_[0])/dt),siz,dt)
          Hi.append(HI3_)
          HI4_=HI4_.reshape(int(len(HI4_[0])/dt),siz,dt)
          Hi.append(HI4_)
          HI5_=HI5_.reshape(int(len(HI5_[0])/dt),siz,dt)
          Hi.append(HI5_)

          lenn=[]
          for i in range(shape(Hi)[0]):
               lenn.append(shape(Hi[i])[0])


          minl=np.min(lenn)

          #SVD on the reservoir activity check how much the principal comp. overlap

          #same numbers
          
          svd_same=[]
          svd_diff=[]
          for x in range(cho+1):
               for y in range(cho+1):
                    voo1=[]
                    voo2=[]
                    for k in range(minl):
                         p=dt-1
                         
                         s1,vt1=linalg.eig(np.dot(Hi[x][k].T,Hi[x][k]))#,p, which='LM')
                         s2,vt2=linalg.eig(np.dot(Hi[y+cho+1][k].T,Hi[y+cho+1][k]))
                         s3,vt3=linalg.eig(np.dot(Hi[y+2*(cho+1)][k].T,Hi[y+3*(cho+1)][k]))#,p, which='LM')
                         
                         #s3,vt3=linalg.eig(np.dot(HI3_.T,HI3_))
                         vt1=vt1.T
                         vt2=vt2.T
                         vt3=vt3.T

                         tot12=[]
                         tot13=[]
                         yo=deepcopy(vt1)
                         for i in range(shape(vt1)[0]):
                              l=linalg.norm(yo[i])
                              for j in range(i+1):
                                   #j==i
                                   yo[i]=yo[i]-vt2[j]*np.dot(yo[i],vt2[j])/linalg.norm(vt2[j])
                              tot12.append(linalg.norm(yo[i])/l)

                         voo1.append(tot12[0])

                         yo=deepcopy(vt1)
                         for i in range(shape(vt1)[0]):
                              l=linalg.norm(yo[i])
                              for j in range(i+1):
                                   #j==i
                                   yo[i]=yo[i]-vt3[j]*np.dot(yo[i],vt3[j])/linalg.norm(vt3[j])
                              tot13.append(linalg.norm(yo[i])/l)

                         voo2.append(tot13[0])
                         #print ('\n\n difftrans', tot12[0:10], '\n\n\n sametran \n\n\n', tot13[0:10])
                    voodoo1=np.mean(voo1)
                    voodoo2=np.mean(voo2)
                    svd_same.append(voodoo1)
                    svd_diff.append(voodoo2)

          
          svd_same=1-np.array(svd_same).reshape(cho+1,cho+1)
          svd_diff=1-np.array(svd_diff).reshape(cho+1,cho+1)
          #end of svd

          np.savetxt('PCA overlap_same.txt', svd_same)

          np.savetxt('PCA overlap_diff.txt', svd_diff)

          print(svd_same, '\n\n',svd_diff, '\n\n')

          Hii=deepcopy(Hi)

          for i in range(shape(Hi)[0]):
               Hii[i]=np.reshape(Hii[i],(1000,dt*shape(Hi[i])[0]))
               Hi[i]=Hi[i].mean(axis=0)'''
               
          
          
          '''Hi0=Hi0.mean(axis=0)
          st=matrix(Hi0).std(0)
          Hi0=Hi0.reshape(siz,dt)

          
          Hi1=Hi1.mean(axis=0)
          st=matrix(Hi1).std(0)
          Hi1=Hi1.reshape(siz,dt)

          
          Hi2=Hi2.mean(axis=0)
          st=matrix(Hi2).std(0)
          Hi2=Hi2.reshape(siz,dt)

          
          Hi3=Hi3.mean(axis=0)
          st=matrix(Hi3).std(0)
          Hi3=Hi3.reshape(siz,dt)

         
          Hi4=Hi4.mean(axis=0)
          st=matrix(Hi4).std(0)
          Hi4=Hi4.reshape(siz,dt)

          
          Hi0_=Hi0_.mean(axis=0)
          st=matrix(Hi0_).std(0)
          Hi0_=Hi0_.reshape(siz,dt)

          
          Hi1_=Hi1_.mean(axis=0)
          st=matrix(Hi1_).std(0)
          Hi1_=Hi1_.reshape(siz,dt)

          
          Hi2_=Hi2_.mean(axis=0)
          st=matrix(Hi2_).std(0)
          Hi2_=Hi2_.reshape(siz,dt)

          
          Hi3_=Hi3_.mean(axis=0)
          st=matrix(Hi3_).std(0)
          Hi3_=Hi3_.reshape(siz,dt)
          
          
          Hi4_=Hi4_.mean(axis=0)
          st=matrix(Hi4_).std(0)
          Hi4_=Hi4_.reshape(siz,dt)

          HI0=HI0.mean(axis=0)
          HI0=HI0.reshape(siz,dt)

          HI1=HI1.mean(axis=0)
          HI1=HI1.reshape(siz,dt)

          HI2=HI2.mean(axis=0)
          HI2=HI2.reshape(siz,dt)

          HI3=HI3.mean(axis=0)
          HI3=HI3.reshape(siz,dt)

          HI4=HI4.mean(axis=0)
          HI4=HI4.reshape(siz,dt)

          HI0_=HI0_.mean(axis=0)
          HI0_=HI0_.reshape(siz,dt)


          HI1_=HI1_.mean(axis=0)
          HI1_=HI1_.reshape(siz,dt)


          HI2_=HI2_.mean(axis=0)
          HI2_=HI2_.reshape(siz,dt)


          HI3_=HI3_.mean(axis=0)
          HI3_=HI3_.reshape(siz,dt)

 
          HI3_=HI3_.mean(axis=0)
          HI3_=HI3_.reshape(siz,dt)'''


          print (Y, X, dt)

          clf = KernelRidge(alpha=1)
          clf.fit(X.T, Y.T) 
          Wpred = dot(dot(Y,X.T), linalg.inv(dot(X,X.T) + err*eye(dot(X,X.T).shape[0]) ) )
          print ('yay')

          '''for s in range(6):
               N=4
               nodes=50
               M=imshow(Hii[s][:nodes,:N*dt],  vmax=abs(X[30:][:]).max(),vmin=-abs(X[30:][:]).max(), interpolation='nearest', aspect='auto', origin='upper')
               cb=colorbar(M,shrink=0.5, pad=.1)
               cb.set_label('Reservoir  Activity', fontsize=10)
               ylabel('$Reservoir  Nodes$', fontsize=13)
               xlabel('$Time $', fontsize=13)
               #title('Reservoir activity', fontsize=20)
               plt.savefig('res_activity' +str(s)+'.jpg',pad_inches=1)
               plt.close()

               W=imshow(Wpred[:,:nodes], interpolation='nearest', aspect='auto', origin='upper')
               yint=[0,1,2,3,4,5]
               plt.yticks(yint)
               cb=colorbar(W,shrink=0.3, pad=.1)
               cb.set_label('Weights', fontsize=10)
               ylabel('$Readout  Nodes$', fontsize=13)
               xlabel('$ Reservoir  Nodes)$', fontsize=13)
               #title('', fontsize=20)
               plt.savefig('Wout'+str(nodes)+'nodes.jpg',pad_inches=1)
               plt.close()


               colors=cm.rainbow(np.linspace(0, 1, len(yint)))
               mymap = mpl.colors.LinearSegmentedColormap.from_list('mycolors',colors)

          for s in range(6):
               print(s, shape(Hii[s]))
               plt.plot(np.linspace(0,dt*N,dt*N),Hii[s][10,:dt*N])#,color=colors[i])
               #plt.plot(np.linspace(0,dt*N,dt*N),X[i,2*boo*dt:dt*(2*boo+N)])#,color=colors[i], label=str(i))
               #plt.plot(np.linspace(0,dt*N,dt*N),X[i,boo*dt:dt*(boo+N)])#,color=colors[i], label=str(i))
               plt.xlabel('$Time$')
               plt.legend(['Very Similar', 'Rotated 90','Rotated 270' , 'Blurred','Zoon','Different'])
               plt.ylabel('$Node Activity $')
               #plt.title('Node '+ str(i))
               plt.savefig('node_activity_for_first' + str(i) + 'nodes.jpg')
               plt.close()'''
          '''sm = plt.cm.ScalarMappable(cmap=mymap, norm=plt.Normalize(vmin=1, vmax=5))
          sm._A = []
          l=colorbar(sm)
          l.set_label('nodes',size=18)


          imp=list([])
          print(type(imp))
          avg=absolute(Wpred).mean()
          for i in range(3):
               imp1=np.argsort(Wpred[i,:])[-20:]
               for j in range(len(imp1)):
                    imp.append(imp1[j])
          imp=unique(imp)


          Hi00=Hi0[imp,:]
          Hi11=Hi1[imp,:]
          Hi22=Hi2[imp,:]

          num=10#dt
          pca = PCA(n_components=num)
          pca.fit(Hi0)
          a=pca.components_
          #print (a, 'a')
          a1=pca.explained_variance_ratio_

          pca2 = PCA(n_components=num)
          pca2.fit(Hi1)
          b=pca2.components_
          b1=pca.explained_variance_ratio_
          #print (b, 'b')

          pca3 = PCA(n_components=num)
          pca3.fit(Hi2)
          c=pca3.components_
          c1=pca.explained_variance_ratio_




          d=zeros((num,num))
          for i in range(num):
               for j in range(num):
                    d[i][j]=np.dot(a[i],b[j])

          e=zeros((num,num))
          for i in range(num):
               for j in range(num):
                    e[i][j]=np.dot(a[i],c[j])





          V=imshow(d,  vmax=abs(0.75),vmin=-abs(0.75), interpolation='nearest', aspect='auto', origin='upper')
          cb=colorbar(V,shrink=0.3, pad=.1)
          ylabel('$eigenv of image b$', fontsize=13)
          xlabel('$eigenvectors of image a$', fontsize=13)
          title('Same: dot product of PCs', fontsize=20)
          plt.savefig('sameandrotateeig100.jpg',pad_inches=1)
          plt.close()

          V=imshow(e,  vmax=abs(0.75),vmin=-abs(0.75), interpolation='nearest', aspect='auto', origin='upper')
          cb=colorbar(V,shrink=0.3, pad=.1)
          ylabel('$eigenv of image b$', fontsize=13)
          xlabel('$eigenvectors of image a$', fontsize=13)
          title('Same: dot product of PCs', fontsize=20)
          plt.savefig('sameandsameeig100.jpg',pad_inches=1)
          plt.close()'''




          #prediction state

          random.seed(None)
          lout=np.where(test_set[1]>5)[0]
          total=0
          for jiggs in range(5):
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

               print('success rate for',jiggs, correct/(i+1))
          total= total/((jiggs+1)*(i+1))
          print ('total success rate and spectr: ',total, cv)
          spectr[cv]=total

with open('performance_with_spectr_rad.json', 'a') as f:
     json.dump(spectr,f)
     f.write('\n')

plt.scatter(list(spectr.keys()),list(spectr.values()))
xlabel('$Spectral \quad Radius$', fontsize=13)
ylabel('$Fraction \quad correct$', fontsize=13)
plt.savefig('performance_v_spectral_radius',pad_inches=1)
plt.close()
     

#find corelation coeff betwe res activity

'''corr_same=[]
corr_diff=[]
for x in range(cho+1):
     for y in range(cho+1):
          corr_same.append(corr2(Hi[x],Hi[y+cho+1]))
          corr_diff.append(corr2(Hi[x],Hi[y+3*(cho+1)]))

corr_same=np.array(corr_same).reshape(cho+1,cho+1)
corr_diff=np.array(corr_diff).reshape(cho+1,cho+1)

print ('/n/n',corr_same,'/n/n', corr_diff)'''










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

g.close()




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
    
    
