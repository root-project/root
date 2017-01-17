#include<Mpi.h>
#include <TVectorT.h>
#include <TVectorDfwd.h>
#include <TVectorD.h>
#include <TMatrixD.h>
#include <cassert>
#include <TSystem.h>
using namespace ROOT::Mpi;



template<class T> T Op(const T &a, const T &b)
{
   return a + b;
}



void reduce_test_scalar(Int_t root = 0)
{
   gSystem->Load("libMatrix");
   auto rank = gComm->GetRank();
   auto size = gComm->GetSize();

   /////////////////////////
   //testing custom object//
   /////////////////////////
//    TMatrixD send_mat(size,size);
   TVectorD send_vec(size);
   if (root == rank) {
      for (auto i = 0; i < size; i++) {
         send_vec[i] = 1;
//        for (auto j = 0; j < size; j++) send_mat[i][j]=1.0;
      }
   }

   Int_t value = 0;
   gComm->Reduce(rank, value, Op<Int_t>, root); //testing custom object
   gComm->Barrier();

//    TMatrixD recv_mat;
//    gComm->Reduce(send_mat, recv_mat, Op<TMatrixD>, root); //testing custom object
//    gComm->Barrier();

//    TVectorD recv_vec(size);
//    gComm->Reduce(send_vec, recv_vec, Op<TVectorD>, root); //testing custom object
//    gComm->Barrier();


   if (rank == root) {
      int sum = 0;
      for (int i = 0; i < size; i++) {
         sum = Op(sum, i);
      }
//       std::cout<<std::endl;
      printf("MPI Result     = %d\n", value);
      printf("Correct Result = %d\n", sum);
//      recv_mat.Print();
//       recv_vec.Print();
   }

   /*
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

      if (rank == root) delete[] send_vec;*/
}

void reduce(Bool_t stressTest = kFALSE)
{
   TEnvironment env;
   if (gComm->GetSize() == 1) return; //needed at least 2 process
//    if (!stressTest)
   reduce_test_scalar(1);
//    else {
//       //stressTest
//       for (auto i = 0; i < gComm->GetSize(); i++)
//          for (auto j = 1; j < gComm->GetSize() + 1; j++) //count can not be zero
//             reduce_test(i, j);
//    }
}


