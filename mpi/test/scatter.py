from ROOT import Mpi, TMatrixD, TString
from ROOT.Mpi import COMM_WORLD, TEnvironment
env=TEnvironment()
def scatter():
    a=[]
    if COMM_WORLD.IsMainProcess():
        mat=TMatrixD(1,1)
        mat[0][0]=123
        a.append(1)
        a.append("rmpi")
        a.append(0.5)
        a.append(mat)
    obj=COMM_WORLD.Scatter(a,4,1,0)
    print(obj)
    if COMM_WORLD.GetRank() == 0:
        assert(obj[0]==1)
    if COMM_WORLD.GetRank() == 1:
        assert(obj[0]=="rmpi")
    if COMM_WORLD.GetRank() == 2:
        assert(obj[0]==0.5)
    if COMM_WORLD.GetRank() == 4:
        assert(obj[0][0][0]==123)


    
if __name__ == "__main__":
    scatter()

