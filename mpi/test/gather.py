from ROOT import Mpi, TMatrixD, TString
from ROOT.Mpi import COMM_WORLD, TEnvironment
env=TEnvironment()
def gather():
    msg=[COMM_WORLD.GetRank()];
    obj=COMM_WORLD.Gather(msg,1,COMM_WORLD.GetSize(),0)
    if COMM_WORLD.IsMainProcess():
        print(obj)
        assert(obj==range(0,COMM_WORLD.GetSize(),1))
    
    msg=[COMM_WORLD.GetRank(),COMM_WORLD.GetRank()];
    obj=COMM_WORLD.Gather(msg,2,COMM_WORLD.GetSize(),0)
    if COMM_WORLD.IsMainProcess():
        print(obj)
        data=[]
        for i in range(0,COMM_WORLD.GetSize(),1):
            data.extend([i,i])
        assert(obj==data)

    
if __name__ == "__main__":
    gather()

