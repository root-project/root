#include <Mpi.h>
#include <TMatrixD.h>
using namespace ROOT::Mpi;
void bcast()
{
   TEnvironment env;

   if (COMM_WORLD.GetSize() == 1) return; // need at least 2 process

   // data to send/recv
   TMatrixD mymat(2, 2); // ROOT object

   auto rank = COMM_WORLD.GetRank();
   auto root = COMM_WORLD.GetMainProcess();
   if (COMM_WORLD.IsMainProcess()) {
      mymat[0][0] = 0.1;
      mymat[0][1] = 0.2;
      mymat[1][0] = 0.3;
      mymat[1][1] = 0.4;
   }

   COMM_WORLD.Bcast(mymat, root);
   std::cout << "Rank = " << rank << std::endl;
   mymat.Print();
   std::cout.flush();
}
