#include <Mpi.h>
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

void read(TString filename = "mpiread.root")
{
   auto f = TMpiFile::Open(COMM_WORLD, filename, "READ");
   auto funct = (TF1 *)f->Get(Form("f%d", COMM_WORLD.GetRank()));
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

void clean(TString filename)
{
   if (COMM_WORLD.GetRank() == 0) {
      if (!gSystem->AccessPathName(filename, kFileExists)) {
         if (gSystem->Unlink(filename) != 0) {
            COMM_WORLD.Abort(ERR_FILE);
         }
      }
   }
}

void file(Int_t size = 10)
{
   TEnvironment env; // environment to start communication system
   //    env.SyncOutput();

   create();
   // TODO: add some extra test here
   clean("mpicreate.root");

   recreate();
   // TODO: add some extra test here
   clean("mpirecreate.root");

   create("mpiupdate.root");
   update();
   // TODO: add some extra test here
   clean("mpiupdate.root");

   create("mpiread.root");
   read();
   // TODO: add some extra test here
   clean("mpiread.root");

   test_sync();
   // TODO: add some extra test here
   clean("mpisync.root");
}
