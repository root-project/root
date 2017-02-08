#include<Mpi.h>
#include<TMatrixD.h>
#include <cassert>
using namespace ROOT::Mpi;


void scatter_test(Int_t root = 0, Int_t count = 2)
{
   auto rank = COMM_WORLD.GetRank();
   auto size = COMM_WORLD.GetSize();

   /////////////////////////
   //testing custom object//
   /////////////////////////
   TVectorD *send_vec;
   if (root == rank) {
      send_vec = new TVectorD[size * count];
      for (auto i = 0; i < COMM_WORLD.GetSize() * count; i++) {
         send_vec[i].ResizeTo(1);
         send_vec[i][0] = i;
      }
   }
   TVectorD recv_vec[count];

   COMM_WORLD.Scatter(send_vec, size * count, recv_vec, count, root); //testing custom object

   std::cout << "--------- Rank = " << rank << std::endl;
   std::cout.flush();
   for (auto i = 0; i < count; i++) {
      recv_vec[i].Print();
      std::cout << recv_vec[i][0] << " -- " << (rank * count + i) << std::endl;
      //assertions
      assert(recv_vec[i][0] == (rank * count + i));
   }
   std::cout << "--------- Rank = " << rank << std::endl;
   std::cout.flush();

   if (rank == root) delete[] send_vec;
}

void scatter(Bool_t stressTest = kTRUE)
{
   TEnvironment env;
   if (COMM_WORLD.GetSize() == 1) return; //needed at least 2 process
   if (!stressTest) scatter_test();
   else {
      //stressTest
      for (auto i = 0; i < COMM_WORLD.GetSize(); i++)
         for (auto j = 1; j < COMM_WORLD.GetSize() + 1; j++) //count can not be zero
            scatter_test(i, j);
   }
}


