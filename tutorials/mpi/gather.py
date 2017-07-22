## \file
## \ingroup tutorial_mpi
##
## Collect messages from a group of processes. 
## Each process (root process included) sends the contents of its send buffer to the root process. 
## The root process receives the messages and stores them in rank order. 
## The outcome is as if each of the n processes in the group (including the root process) had executed a call to ROOT::Mpi:TCommunicator::Gather
## to execute this example with 4 processors, do:
##
## ~~~{.cpp}
##  rootmpi -np 4 gather.py
## ~~~
##
##
## \macro_output
## \macro_code
##
## \author Omar Zapata


from ROOT import Mpi, TVectorD
from ROOT.Mpi import TEnvironment, COMM_WORLD

def gather():
    
   env=TEnvironment()
   if COMM_WORLD.GetSize() == 1 :   return; # needed at least 2 process
   rank = COMM_WORLD.GetRank()
   size = COMM_WORLD.GetSize()

   count = 2;
   root = COMM_WORLD.GetMainProcess();

   #creating a vector to send 
   send_vec=[]
   for i in range(0,count,1):
      vec=TVectorD(1)
      vec[0]=rank
      send_vec.append(vec)


   recv_vec = COMM_WORLD.Gather(send_vec, count, size * count, root) 

   if rank == root :
      # just printing all infortaion
      for i in range(0,size * count,1):
         recv_vec[i].Print()

      for i in range(0,COMM_WORLD.GetSize(),1):
         for j in range(0,count,1):
            print("vec[%i] = %f -- %i"%(i * count + j ,recv_vec[i * count + j][0] ,i ))

if __name__ == "__main__":
    gather()
