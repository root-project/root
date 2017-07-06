#include <TMpi.h>
#include <cassert>
using namespace ROOT::Mpi;

void env(Int_t level = THREAD_SINGLE)
{
   TEnvironment env(level); // environment to start communication system

   assert(env.IsFinalized() == kFALSE);
   assert(env.IsInitialized() == kTRUE);
   assert(env.GetThreadLevel() == level);

   assert(env.IsMainThread() == kTRUE);

   // testing synchronized output
   env.SyncOutput();

   assert(env.IsSyncOutput() == kTRUE);

   // testing with std::ostream
   std::cout << Form("CoutRank%d", COMM_WORLD.GetRank()) << std::flush;
   std::cerr << Form("CerrRank%d", COMM_WORLD.GetRank()) << std::flush;
   env.SyncOutput(kFALSE);

   assert(env.IsSyncOutput() == kFALSE);

   assert(env.GetStdOut() == Form("CoutRank%d", COMM_WORLD.GetRank()));
   assert(env.GetStdErr() == Form("CerrRank%d", COMM_WORLD.GetRank()));

   env.SyncOutput();
   // testing with FILE pointer
   fprintf(stdout, "%s", Form("StdoutRank%d", COMM_WORLD.GetRank()));
   fprintf(stderr, "%s", Form("StderrRank%d", COMM_WORLD.GetRank()));

   env.SyncOutput(kFALSE);

   assert(env.GetStdOut() == Form("StdoutRank%d", COMM_WORLD.GetRank()));
   assert(env.GetStdErr() == Form("StderrRank%d", COMM_WORLD.GetRank()));

   env.SyncOutput();
   std::cout << Form("CRank%d", COMM_WORLD.GetRank()) << std::flush;
   std::cerr << Form("ERank%d", COMM_WORLD.GetRank()) << std::flush;
   env.EndCapture();

   assert(env.GetStdOut() == Form("CRank%d", COMM_WORLD.GetRank()));
   assert(env.GetStdErr() == Form("ERank%d", COMM_WORLD.GetRank()));
   env.ClearBuffers();
   assert(env.GetStdOut() == "");
   assert(env.GetStdErr() == "");

   env.Finalize();
   assert(env.IsFinalized() == kTRUE);
}

Int_t main()
{
   env();
   return 0;
}
