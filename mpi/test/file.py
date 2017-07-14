from ROOT import Mpi, TMatrixD, TString, gSystem, kFileExists, Form, TF1
from ROOT.Mpi import COMM_WORLD, TEnvironment, TMpiFile
import numpy as np
env=TEnvironment()

def create(filename = "mpicreate.root"):

   if gSystem.AccessPathName(filename, kFileExists) == False:
      if gSystem.Unlink(filename) != 0:
         COMM_WORLD.Abort(Mpi.ERR_FILE)

   f = TMpiFile.Open(COMM_WORLD, filename, "CREATE")
   f1=TF1("f%i"%COMM_WORLD.GetRank(), "%d*sin(x)"%(COMM_WORLD.GetRank() + 1))
   f1.Write()
   f.Save()
   f.Close()
   del f

def recreate(filename = "mpirecreate.root"):
   f = TMpiFile.Open(COMM_WORLD, filename, "RECREATE")
   f1=TF1("f%i"%COMM_WORLD.GetRank(), "%d*sin(x)"%(COMM_WORLD.GetRank() + 1))
   f1.Write()
   f.Save()
   f.Close()
   del f

def update(filename = "mpiupdate.root"):
   f = TMpiFile.Open(COMM_WORLD, filename, "UPDATE")
   f1=TF1("f%i"%COMM_WORLD.GetRank(), "%d*sin(x)"%(COMM_WORLD.GetRank() + 1))
   f1.Write()
   f.Save()
   f.Close()
   del f

def read(filename = "mpiread.root"):
   f = TMpiFile.Open(COMM_WORLD, filename, "READ")
   funct = f.Get("f%d"%COMM_WORLD.GetRank())
   f.Close()
   del f

def test_sync(filename = "mpisync.root"):
   f = TMpiFile.Open(COMM_WORLD, filename, "RECREATE")
   f1=TF1("f%i"%COMM_WORLD.GetRank(), "%d*sin(x)"%(COMM_WORLD.GetRank() + 1))
   f1.Write()
   f.Save()
   f.Close()
   del f

def clean(filename):
   if COMM_WORLD.GetRank() == 0:
      if gSystem.AccessPathName(filename, kFileExists) == False:
         if gSystem.Unlink(filename) != 0 :
            COMM_WORLD.Abort(Mpi.ERR_FILE)

def mpifile(size = 10):
   env.SyncOutput();

   create();
   #TODO: add some extra test here
   clean("mpicreate.root");

   recreate();
   #TODO: add some extra test here
   clean("mpirecreate.root");

   create("mpiupdate.root");
   update();
   # TODO: add some extra test here
   clean("mpiupdate.root");

   create("mpiread.root");
   read();
   # TODO: add some extra test here
   clean("mpiread.root");

   test_sync();
   # TODO: add some extra test here
   clean("mpisync.root");

if __name__ == "__main__":
    mpifile()
