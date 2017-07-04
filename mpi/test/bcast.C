#include <Mpi.h>
#include <TMatrixD.h>
#include <cassert>
#include "particle.h"
using namespace ROOT::Mpi;

void bcast_test_scalar(Int_t root = 0, Int_t size = 2)
{

   auto rank = COMM_WORLD.GetRank();

   //////////////////////////
   // testing TMpiMessage  //
   /////////////////////////
   TMpiMessage msg;
   if (rank == root) {
      TMatrixD mymat(size, size); // ROOT object
      mymat[0][0] = 0.1;
      mymat[0][1] = 0.2;
      mymat[1][0] = 0.3;
      mymat[1][1] = 0.4;
      msg.WriteObject(mymat);
   }

   COMM_WORLD.Bcast(msg, root); // testing TMpiMessage
   auto mat = (TMatrixD *)msg.ReadObjectAny(TMatrixD::Class());

   std::cout << "Rank = " << rank << std::endl;
   // mat->Print();
   std::cout.flush();
   TMatrixD req_mat(size, size);
   req_mat[0][0] = 0.1;
   req_mat[0][1] = 0.2;
   req_mat[1][0] = 0.3;
   req_mat[1][1] = 0.4;

   /////////////////////////
   // testing custom object//
   /////////////////////////
   Particle<Int_t> p;
   if (rank == root) {
      p.Set(1, 2); // if root process fill the particle
   }
   COMM_WORLD.Bcast(p, root); // testing custom object
   // p.Print();

   // assertions
   assert((*mat)[0][0] == req_mat[0][0]);
   assert((*mat)[0][1] == req_mat[0][1]);
   assert((*mat)[1][0] == req_mat[1][0]);
   assert((*mat)[1][1] == req_mat[1][1]);
   assert(p.GetX() == 1);
   assert(p.GetY() == 2);
}

void bcast_test_array(Int_t root = 0, Int_t size = 2, Int_t count = 4)
{

   auto rank = COMM_WORLD.GetRank();
   TVectorD vecs[count];
   Int_t arr[count];
   if (root == rank) {
      for (auto i = 0; i < count; i++) {
         vecs[i].ResizeTo(count);
         vecs[i][0] = 1.0;
         arr[i] = i;
      }
   }
   COMM_WORLD.Bcast(vecs, count, root);
   COMM_WORLD.Bcast(arr, count, root);
   for (auto i = 0; i < count; i++) {
      vecs[i].Print();
      assert(vecs[i][0] == 1.0);
      assert(arr[i] == i);
   }
}

// void bcast(Bool_t stressTest = kTRUE)
void bcast(Bool_t stressTest = kFALSE)
{
   TEnvironment env;
   if (COMM_WORLD.GetSize() == 1) return; // needed at least 2 process
   bcast_test_scalar();
   if (!stressTest) {
      bcast_test_scalar();
      bcast_test_array();
   } else {
      // stressTest
      for (auto i = 0; i < COMM_WORLD.GetSize(); i++)
         for (auto j = 1; j < COMM_WORLD.GetSize() + 1; j++) // count can not be zero
            bcast_test_scalar(i, j * 100);
   }
}
