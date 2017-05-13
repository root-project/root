#include<Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;

void file(Int_t size = 10)
{
   TEnvironment env;          //environment to start communication system
//    env.SyncOutput();

   auto f = TMpiFile::Open(COMM_WORLD, "mpifile.root", "RECREATE");

   auto rank = COMM_WORLD.GetRank();
   TF1 f1(Form("f%d", COMM_WORLD.GetRank()), Form("%d*sin(x)", rank + 1));
   f1.Write();
//
// //    f.Merge(0, TFileMerger::kAllIncremental);
//    if (rank == 0) {
//       TF1 fs("fspecial", Form("sin(x)/%d", rank + 1));
//       fs.Write();
// //        f.ls();
//    }
//    f.Merge(0, TFileMerger::kAllIncremental);

//    f.Sync();
//    f.Merge();
//    f.ls();
   f->Save();
//    if (rank == 0 || rank == 1 || rank == 2) f.Save();

   f->Sync();
   f->ls();
   f->Close();
}
