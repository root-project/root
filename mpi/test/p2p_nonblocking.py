from ROOT import TMatrixD, Mpi
from ROOT.Mpi import COMM_WORLD, TEnvironment
env=TEnvironment()


def ip2p():
   
   if COMM_WORLD.IsMainProcess(): 
      mymat=TMatrixD(2, 2);                     # ROOT object
      mymat[0][0] = 0.1;
      mymat[0][1] = 0.2;
      mymat[1][0] = 0.3;
      mymat[1][1] = 0.4;

      print("Sending first mat = ")
      mymat.Print();
      req = COMM_WORLD.ISend(mymat, 1, 2);
      req.Wait();

      print("Sending second list = ")
      l=[1,2,3]
      print(l)
      req = COMM_WORLD.ISend(l, 1, 3);
      req.Wait();

   else:
      req = COMM_WORLD.IRecv(0, 3);
      req.Wait();
      obj=req.GetMsg()
      print("Received first list = ")
      print(obj)
      assert(obj==[1,2,3]) 
       
      req = COMM_WORLD.IRecv(0, 2);
      req.Wait();
      print("Received second object mat = ")
      mat=req.GetMsg()
      mat.Print();
      assert(mat[0][0] == 0.1);
      assert(mat[0][1] == 0.2);
      assert(mat[1][0] == 0.3);
      assert(mat[1][1] == 0.4);
      
if __name__ == "__main__":
    ip2p()
