#include<Mpi.h>
#include<TMatrixD.h>
#include <cassert>
using namespace ROOT::Mpi;


void scatter()
{
   TEnvironment env;

   if (gComm->GetSize() == 1) return; //needed at least 2 process

   auto rank = gComm->GetRank();
   auto size = gComm->GetSize();
   auto count = 2;
   auto root = 0;

   /////////////////////////
   //testing custom object//
   /////////////////////////
   TVectorD send_vec[size * count];
   if (root == rank) {
      for (auto i = 0; i < gComm->GetSize() * count; i++) {
         send_vec[i].ResizeTo(1);
         send_vec[i][0] = i;
      }
   }
   TVectorD recv_vec[count];

   gComm->Scatter(send_vec, size * count, recv_vec, count, root); //testing custom object

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
}


