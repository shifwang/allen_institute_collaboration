import sklearn
from multiprocessing import Pool
a = Pool(2)
def f(i):
    print(sklearn.utils.resample([1,2,3,4,5], random_state = i)) 
a.map(f, range(20))
   
