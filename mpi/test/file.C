#include<Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;

void file(Int_t size = 10)
{
   TEnvironment env;          //environment to start communication system
//    env.SyncOutput();
   TMpiFile f(COMM_WORLD, "mpifile.root", "RECREATE");

   auto rank = COMM_WORLD.GetRank();
   TF1 f1(Form("f%d", COMM_WORLD.GetRank()), Form("%d*sin(x)", rank));
   f1.Write();

   f.Merge(0);
   f.Close();
}
