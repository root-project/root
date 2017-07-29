#include <TMpi.h>
using namespace ROOT::Mpi;

void env(Int_t level = THREAD_SINGLE)
{
   TEnvironment env(level); // environment to start communication system

   ROOT_MPI_ASSERT(env.IsFinalized() == kFALSE);
   ROOT_MPI_ASSERT(env.IsInitialized() == kTRUE);
   ROOT_MPI_ASSERT(env.GetThreadLevel() == level);

   ROOT_MPI_ASSERT(env.IsMainThread() == kTRUE);

   // testing synchronized output
   env.SyncOutput();

   ROOT_MPI_ASSERT(env.IsSyncOutput() == kTRUE);

   // testing with std::ostream
   std::cout << Form("CoutRank%d", COMM_WORLD.GetRank()) << std::flush;
   std::cerr << Form("CerrRank%d", COMM_WORLD.GetRank()) << std::flush;
   env.SyncOutput(kFALSE);

   ROOT_MPI_ASSERT(env.IsSyncOutput() == kFALSE);

   ROOT_MPI_ASSERT(env.GetStdOut() == Form("CoutRank%d", COMM_WORLD.GetRank()));
   ROOT_MPI_ASSERT(env.GetStdErr() == Form("CerrRank%d", COMM_WORLD.GetRank()));

   env.SyncOutput();
   // testing with FILE pointer
   fprintf(stdout, "%s", Form("StdoutRank%d", COMM_WORLD.GetRank()));
   fprintf(stderr, "%s", Form("StderrRank%d", COMM_WORLD.GetRank()));

   env.SyncOutput(kFALSE);

   ROOT_MPI_ASSERT(env.GetStdOut() == Form("StdoutRank%d", COMM_WORLD.GetRank()));
   ROOT_MPI_ASSERT(env.GetStdErr() == Form("StderrRank%d", COMM_WORLD.GetRank()));

   env.SyncOutput();
   std::cout << Form("CRank%d", COMM_WORLD.GetRank()) << std::flush;
   std::cerr << Form("ERank%d", COMM_WORLD.GetRank()) << std::flush;
   env.EndCapture();

   ROOT_MPI_ASSERT(env.GetStdOut() == Form("CRank%d", COMM_WORLD.GetRank()));
   ROOT_MPI_ASSERT(env.GetStdErr() == Form("ERank%d", COMM_WORLD.GetRank()));
   env.ClearBuffers();
   ROOT_MPI_ASSERT(env.GetStdOut() == "");
   ROOT_MPI_ASSERT(env.GetStdErr() == "");

   env.Finalize();
   ROOT_MPI_ASSERT(env.IsFinalized() == kTRUE);
}

Int_t main()
{
   env();
   return 0;
}
