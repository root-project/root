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
