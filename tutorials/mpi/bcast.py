## \file
## \ingroup tutorial_mpi
##
## Broadcasts a message from the process with rank root to all processes of the group, itself included.
## It is called by all members of group using the same arguments for comm, root.
## On return, the contents of root’s communication buffer has been copied to all processes.
## to execute this example with 4 processors, do:
##
## ~~~{.cpp}
##  rootmpi -np 4 bcast.py
## ~~~
##
##
## \macro_output
## \macro_code
##
## \author Omar Zapata


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
