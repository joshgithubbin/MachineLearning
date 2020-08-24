import pylab
import numpy as np
from sklearn.cluster import KMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import silhouette_score
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV


pylab.rcParams["font.family"] = "Times New Roman"


###############################################################################

def kmean_optimize(points,kmax,ins='random',itrs=12,rs=1,plot='yes',save='no'):
    
    sse = []
    sil = []
    k_values = np.arange(2,kmax+1,1)
    
    
    for k in range(2,kmax+1):
        k_means = KMeans(n_clusters=k,init='k-means++',random_state=rs).fit(points)
        
        #elbow method
            
        sse.append(k_means.inertia_)
        
        #silhouette method
        
        labels = k_means.labels_
        score = silhouette_score(points, labels, metric = 'euclidean')
        sil.append(score)
        
        
    if plot == 'yes':
        
        #plot k vs sse
        #pylab.subplot(2,1,1)
        pylab.plot(k_values,sse,color='blue')
        pylab.title('Inertia vs. Number of Clusters',size=20,weight='bold')
        pylab.ylabel('Inertia',size=15)
        pylab.xlabel('Number of Clusters K',size=15)
        pylab.yticks(np.arange(0,62,step=10))
        pylab.grid(True)
        pylab.savefig('Inertia.png',dpi=1000)
        
        
        pylab.show()
        #plot k vs sil
        #pylab.subplot(2,1,2)
        pylab.plot(k_values,sil,color='red')
        pylab.title('Silhouette Score vs. Number of Clusters', size=20,weight='bold')
        pylab.xlabel('Number of Clusters K', size=15)
        pylab.ylabel('Silhouette Score', size=15)
        pylab.yticks(np.arange(0.5,1.0,step=0.1))
        pylab.grid(True)
        if save != 'no': pylab.savefig('Silhouette-Score.png', dpi=1000)

        pylab.show()
    
    return sse, sil, np.argmax(sil)+2

###############################################################################
    
def knn_optimize(x,y,kmax,ts=0.3,plot='yes',save='no'):
    
    acc = []
    acc_std = []
    err = []
    
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size=ts,random_state=1)    
    
    for k in range(1,kmax):
        
        knn = KNeighborsClassifier(n_neighbors = k).fit(xtrain,ytrain)
        
        ypred = knn.predict(xtest)
        
        acc.append(metrics.accuracy_score(ytest,ypred))
        acc_std.append(np.std(ypred==ytest)/np.sqrt(ypred.shape[0]))
        err.append(np.mean(ypred != ytest))
        
        if plot =='yes':
            pylab.plot(range(1,kmax), acc,color='red')
            pylab.fill_between(range(1,kmax),acc - 1 *acc_std,acc + 1 *acc_std,color='blue',alpha=0.15)
            pylab.ylabel('Accuracy')
            pylab.xlabel('Number of Neighbours K')
            pylab.legend(('Accuracy','+/- STDev'))
            
            if save !='no':pylab.savefig('Accuracy.png')
            
            pylab.plot(range(1,kmax), err,color='red')
            pylab.ylabel('Error Rate')
            pylab.xlabel('Number of Neighbours K')
            
            if save!='no':pylab.savefig('Error-rate.png')
            
    return acc, acc_std, err, np.argmax(acc)+1
        
###############################################################################

def logreg_optimize(x,y,c=(-4,4,5),ts=0.3):
    
    xtrain,xtest,ytrain,ytest = train_test_split(x,y,test_size = ts)
        
    grid = {"C":np.logspace(c[0],c[1],c[2]),"penalty":["l1","l2"]}
    
    lrg = LogisticRegression()
    lrg_cv = GridSearchCV(lrg,grid)
    
    lrg_cv.fit(xtrain,ytrain)
     
    c_b = lrg_cv.best_params_["C"]
    penb = lrg_cv.best_params_["penalty"]
    
    blrg = LogisticRegression(C=c_b,penalty=penb)
    
    blrg.fit(xtrain,ytrain)
    
    score = lrg.score(xtest,ytest)      
    
    return lrg, score, lrg_cv        
        
        