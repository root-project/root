from ROOT import Mpi, TMatrixD
from ROOT.Mpi import TEnvironment, COMM_WORLD
def bcast():
   env=TEnvironment()
   env.SyncOutput()

   if COMM_WORLD.GetSize() == 1:    return; # need at least 2 process


   rank = COMM_WORLD.GetRank();
   root = COMM_WORLD.GetMainProcess();
   # data to send/recv
   mymat=TMatrixD(2, 2); # ROOT object
   if COMM_WORLD.IsMainProcess() :
       
      mymat[0][0] = 0.1;
      mymat[0][1] = 0.2;
      mymat[1][0] = 0.3;
      mymat[1][1] = 0.4;
 
   mat=COMM_WORLD.Bcast(mymat,root)
   print("Rank = %i"%rank)
   mat.Print()

if __name__ == "__main__":
    bcast()
