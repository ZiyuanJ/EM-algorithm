
class Online:
     
    def __init__(self,b):
        self.b = b## learning rate    
    def StaticExpert(self, X,labels):
        # n = number of data-points, d = dimension of data points        
        n, d = X.shape
        p=np.repeat(1/d,d)
        Track=[]
        Loss=[]
        for n in range(n):
            for d in range(d):
                L=X[n,d]-labels[n]
                loss=np.asscalar(L*L)
                p[d]=p[d]*np.exp(-(self.b)*loss)
            p=p/np.sum(p)
            yhat=np.sum(X[n,:]*p)
            Track.append(np.asscalar((yhat-labels[n])*(yhat-labels[n])))
            Loss.append(np.mean(Track))
  
        
        self.params = namedtuple('params', ['L'])
        self.params.L =  Loss
        return self.params       
    
    def plot_Loss(self):
         plt.plot(self.params.L)
         plt.title('Loss vs iteration plot')
         plt.xlabel('Iterations')
         plt.ylabel('Average Loss')
         plt.show()
    

def gettingdata(path,filename):
    os.chdir(path)
    out=[]
    with open(filename) as csvDataFile:
         csvReader = csv.reader(csvDataFile)
         for row in csvReader:
              out.append(row)
    return(np.array(out,dtype='f'))

path='C:\\Users\\Jessie\\SkyDrive\\2017Fall'
filename='cloud.csv'
clouddata=gettingdata(path,filename)

labels=clouddata[:,[6]]
cloud = np.delete(clouddata, 6, 1)

def show(X,labels,b):
    online = Online(b)
    params=online.StaticExpert(X, labels)
    online.plot_Loss( )
   
def superimpose(X,labels,b1,b2,b3,b4):
    params1=Online(b1).StaticExpert(X, labels)
    params2=Online(b2).StaticExpert(X, labels)
    params3=Online(b3).StaticExpert(X, labels)
    params4=Online(b4).StaticExpert(X, labels)
    plt.plot(params1.L)
    plt.plot(params2.L)
    plt.plot(params3.L)
    plt.plot(params4.L)
    plt.legend(['b=%d'%b1,'b=%1.1f'%b2,'b=%1.2f'%b3,'b=%1.3f'%b4], loc='lower right')
    plt.title('Loss vs iteration plot with different learning rate')
    plt.xlabel('Iterations')
    plt.ylabel('Average Loss')
    plt.show()
    
show(cloud,labels,1)
superimpose(cloud,labels,1,0.1,.01,.009)
