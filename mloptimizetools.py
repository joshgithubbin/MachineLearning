import pylab
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
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
        pylab.savefig('Silhouette Score.png', dpi=1000)
        
        if save != 'no':
            pylab.savefig(save, dpi = 1000)
        
    
        pylab.show()
    
    return sse, sil, np.argmax(sil)+2

###############################################################################