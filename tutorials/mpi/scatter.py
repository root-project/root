## \file
## \ingroup tutorial_mpi
##
## Sends data from one task to all tasks in a group.
## This is the inverse operation to ROOT::Mpi::TCommunicator::Gather. 
## An alternative description is that the root sends a message with ROOT::Mpi::TCommunicator::Send. 
## This message is split into n equal segments, the ith segment is sent to the ith process in the group, and each process receives this message as above.
## The send buffer is ignored for all nonroot processes.
## to execute this example with 4 processors, do:
##
## ~~~{.cpp}
##  rootmpi -np 4 scatter.py
## ~~~
##
##
## \macro_output
## \macro_code
##
## \author Omar Zapata

from ROOT import Mpi, TVectorD
from ROOT.Mpi import TEnvironment, COMM_WORLD

def scatter():
    
   env=TEnvironment()
   env.SyncOutput()
   if COMM_WORLD.GetSize() == 1:    return # needed at least 2 process
   rank = COMM_WORLD.GetRank()
   size = COMM_WORLD.GetSize()

   count = 2;
   root = COMM_WORLD.GetMainProcess()

   #creating a vector to send 
   send_vec=[]
   if root == rank:
      #send_vec = new TVectorD[size * count];
      for i in range(0,size * count,1):
         vec=TVectorD(1)
         vec[0]=i;
         send_vec.append(vec)
         

   recv_vec=COMM_WORLD.Scatter(send_vec, size * count, count, root) 

   for i in range(0,count,1):
      recv_vec[i].Print();
      print("%f -- %i"%(recv_vec[i][0] , (rank * count + i) ))

if __name__ == "__main__":
    scatter()
