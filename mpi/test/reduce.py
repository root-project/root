from ROOT import Mpi, TMatrixD, TString
from ROOT.Mpi import COMM_WORLD, TEnvironment
import numpy as np
env=TEnvironment()

def op(obj1,obj2):
    return obj1+obj2

def mreduce():
    msg=COMM_WORLD.GetRank()
    out=COMM_WORLD.Reduce(msg,op,0)
    
    if COMM_WORLD.IsMainProcess():
        print("result=%s"%out)
        #doing the comparison with python builting function
        assert(out==reduce(op,range(0,COMM_WORLD.GetSize())))

    msg=np.array([COMM_WORLD.GetRank(),COMM_WORLD.GetRank()])
    out=COMM_WORLD.Reduce(msg,op,0)
    
    if COMM_WORLD.IsMainProcess():
        print("result=%s"%out)
        
        #doing the comparison with python builting function
        rvalue=reduce(op,range(0,COMM_WORLD.GetSize()))
        assert(out[0]==rvalue)
        assert(out[1]==rvalue)

    msg=COMM_WORLD.GetRank()
    out=COMM_WORLD.AllReduce(msg,op)#return the reduce operation to all ranks
    print("result=%s"%out)
    assert(out==reduce(op,range(0,COMM_WORLD.GetSize())))
    
if __name__ == "__main__":
    mreduce()

