#include<Mpi.h>
#include <TVectorT.h>
#include <TVectorDfwd.h>
#include <TVectorD.h>
#include <TMatrixD.h>
#include <cassert>
#include <TSystem.h>
using namespace ROOT::Mpi;

void reduce_test_scalar(Int_t root = 0)
{
   auto rank = gComm->GetRank();
   auto size = gComm->GetSize();

   /////////////////////////
   //testing custom object//
   /////////////////////////
   TMatrixD send_mat(size, size);
   TVectorD send_vec(size);
   for (auto i = 0; i < size; i++) {
      send_vec[i] = 1;
      for (auto j = 0; j < size; j++) send_mat[i][j] = 1.0;
   }

   Int_t value = 0;

   gComm->Reduce(rank, value, SUM, root); //testing custom object
   gComm->Barrier();

   TMatrixD recv_mat;
   gComm->Reduce(send_mat, recv_mat, SUM, root); //testing custom object
   gComm->Barrier();

   TVectorD recv_vec(size);
   gComm->Reduce(send_vec, recv_vec, SUM, root); //testing custom object
   gComm->Barrier();


   if (rank == root) {
      int sum = 0;
      for (int i = 0; i < size ; i++) {
         sum = SUM<Int_t>()(sum, i);
      }
      std::cout << std::endl;
      printf("MPI Result     = %d\n", value);
      printf("Correct Result = %d\n", sum);
      recv_mat.Print();
      recv_vec.Print();
   }
}

void reduce(Bool_t stressTest = kFALSE)
{
   TEnvironment env;
   if (gComm->GetSize() == 1) return; //needed at least 2 process
   if (!stressTest)   reduce_test_scalar(1);
   else {
      //stressTest
      for (auto i = 0; i < gComm->GetSize(); i++) reduce_test_scalar(i);
   }
}


