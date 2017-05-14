#include<Mpi.h>
#include <cassert>
using namespace ROOT::Mpi;

void create(TString filename = "mpicreate.root")
{
   if (!gSystem->AccessPathName(filename, kFileExists)) {
      if (gSystem->Unlink(filename) != 0) {
         COMM_WORLD.Abort(ERR_FILE);
      }
   }
   auto f = TMpiFile::Open(COMM_WORLD, filename, "CREATE");
   TF1 f1(Form("f%d", COMM_WORLD.GetRank()), Form("%d*sin(x)", COMM_WORLD.GetRank() + 1));
   f1.Write();
   f->Save();
   f->Close();
   delete f;
}

void recreate(TString filename = "mpirecreate.root")
{
   auto f = TMpiFile::Open(COMM_WORLD, filename, "RECREATE");
   TF1 f1(Form("f%d", COMM_WORLD.GetRank()), Form("%d*sin(x)", COMM_WORLD.GetRank() + 1));
   f1.Write();
   f->Save();
   f->Close();
   delete f;
}

void update(TString filename = "mpiupdate.root")
{
   auto f = TMpiFile::Open(COMM_WORLD, filename, "UPDATE");
   TF1 f1(Form("f%d", COMM_WORLD.GetRank()), Form("%d*sin(x)", COMM_WORLD.GetRank() + 1));
   f1.Write();
   f->Save();
   f->Close();
   delete f;
}

void test_sync(TString filename = "mpisync.root")
{
   auto f = TMpiFile::Open(COMM_WORLD, filename, "RECREATE");
   TF1 f1(Form("f%d", COMM_WORLD.GetRank()), Form("%d*sin(x)", COMM_WORLD.GetRank() + 1));
   f1.Write();
   f->Save();
   f->Close();
   delete f;
}

void file(Int_t size = 10)
{
   TEnvironment env;          //environment to start communication system
//    env.SyncOutput();

   create();
   //TODO: add some extra test here
   if (!gSystem->AccessPathName("mpicreate.root", kFileExists)) {
      if (gSystem->Unlink("mpicreate.root") != 0) {
         COMM_WORLD.Abort(ERR_FILE);
      }
   }

   TDirectory::TContext ctxt0;
   recreate();
   //TODO: add some extra test here
   if (!gSystem->AccessPathName("mpirecreate.root", kFileExists)) {
      if (gSystem->Unlink("mpirecreate.root") != 0) {
         COMM_WORLD.Abort(ERR_FILE);
      }
   }

   TDirectory::TContext ctxt1;
   update();
   //TODO: add some extra test here
   if (!gSystem->AccessPathName("mpiupdate.root", kFileExists)) {
      if (gSystem->Unlink("mpiupdate.root") != 0) {
         COMM_WORLD.Abort(ERR_FILE);
      }
   }

   TDirectory::TContext ctxt2;
   test_sync();
   //TODO: add some extra test here
   if (!gSystem->AccessPathName("mpisync.root", kFileExists)) {
      if (gSystem->Unlink("mpisync.root") != 0) {
         COMM_WORLD.Abort(ERR_FILE);
      }
   }

   //    auto f = TMpiFile::Open(COMM_WORLD, "mpifile.root", "UPDATE");
//
//    auto rank = COMM_WORLD.GetRank();
//    TF1 f1(Form("f%d", COMM_WORLD.GetRank()), Form("%d*sin(x)", rank + 1));
//    f1.Write();
// //
// //    f->Merge(0);
// //    f->Save();
//    if (rank == 0) {
//       TF1 fs("fspecial", Form("sin(x)/%d", rank + 1));
//       fs.Write();
// //        f->ls();
//    }
// //    f.Merge(0, TFileMerger::kAllIncremental);
//
//    f->Sync();
// //    f->Merge(0);
// //    f.ls();
// //    f->Save();
//    f->SyncSave();
//
// //    if (rank%2 == 0 ) f->Save();
//
// //    f->Sync();
//    f->ls();
//    f->Close();
}
