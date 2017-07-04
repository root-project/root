#include <Mpi.h>
#include <TMatrixD.h>
#include <cassert>
using namespace ROOT::Mpi;

void gather_test(Int_t root = 0, Int_t count = 2)
{
   auto rank = COMM_WORLD.GetRank();
   auto size = COMM_WORLD.GetSize();

   /////////////////////////
   // testing custom object//
   /////////////////////////
   TVectorD send_vec[count];
   TVectorD *recv_vec;
   for (auto i = 0; i < count; i++) {
      send_vec[i].ResizeTo(1);
      send_vec[i][0] = rank;
   }

   if (rank == root) {
      recv_vec = new TVectorD[size * count];
   }

   COMM_WORLD.Gather(send_vec, count, recv_vec, size * count, root); // testing custom object

   if (rank == root) {
      // just printing all infortaion
      for (auto i = 0; i < size * count; i++) {
         recv_vec[i].Print();
      }

      for (auto i = 0; i < COMM_WORLD.GetSize(); i++) {

         for (auto j = 0; j < count; j++) {
            // assertions
            std::cout << "vec[" << i * count + j << "] = " << recv_vec[i * count + j][0] << " -- " << i << std::endl;
            assert(recv_vec[i * count + j][0] == i);
         }
      }
      delete[] recv_vec;
   }
}

void gather(Bool_t stressTest = kTRUE)
{
   TEnvironment env;
   if (COMM_WORLD.GetSize() == 1) return; // needed at least 2 process
   if (!stressTest)
      gather_test();
   else {
      // stressTest
      for (auto i = 0; i < COMM_WORLD.GetSize(); i++)
         for (auto j = 1; j < COMM_WORLD.GetSize() + 1; j++) // count can not be zero
            gather_test(i, j);
   }
}
