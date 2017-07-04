#include <Mpi.h>
#include <TVectorT.h>
#include <TVectorDfwd.h>
#include <TVectorD.h>
#include <TMatrixD.h>
#include <TComplex.h>
#include <cassert>
#include <TSystem.h>
using namespace ROOT::Mpi;

// TODO:added test for other operators PROD MIN MAX etc..

void reduce_test_scalar(Int_t root = 0)
{
   auto rank = COMM_WORLD.GetRank();
   auto size = COMM_WORLD.GetSize();

   /////////////////////////
   // testing custom object//
   /////////////////////////
   TMatrixD send_mat(size, size);
   TVectorD send_vec(size);
   TComplex send_c(1, 1);
   for (auto i = 0; i < size; i++) {
      send_vec[i] = 1;
      for (auto j = 0; j < size; j++) send_mat[i][j] = 1.0;
   }

   Int_t value = 0;

   COMM_WORLD.Reduce(rank, value, SUM, root); // testing custom object
   COMM_WORLD.Barrier();

   TMatrixD recv_mat;
   COMM_WORLD.Reduce(send_mat, recv_mat, SUM, root); // testing custom object
   COMM_WORLD.Barrier();

   TVectorD recv_vec;
   COMM_WORLD.Reduce(send_vec, recv_vec, SUM, root); // testing custom object
   COMM_WORLD.Barrier();

   TComplex c_value;
   COMM_WORLD.Reduce(send_c, c_value, PROD, root);
   COMM_WORLD.Barrier();

   Int_t prod_value = 0;
   Int_t prod_send = rank + 1;                           // if rank==0 Then it does not make sense
   COMM_WORLD.Reduce(prod_send, prod_value, PROD, root); // testing custom object
   COMM_WORLD.Barrier();

   Int_t min_value;
   COMM_WORLD.Reduce(rank, min_value, MIN, root); // testing custom object
   COMM_WORLD.Barrier();

   Int_t max_value;
   COMM_WORLD.Reduce(rank, max_value, MAX, root); // testing custom object
   COMM_WORLD.Barrier();

   if (rank == root) {
      Int_t sum = 0;
      Int_t prod = 1;
      TComplex cplx_r(1, 1);
      for (Int_t i = 0; i < size - 1; i++) {
         TComplex c_tmp(1, 1);
         cplx_r = PROD<TComplex>()(cplx_r, c_tmp);
      }
      for (Int_t i = 0; i < size; i++) {
         sum = SUM<Int_t>()(sum, i);
         prod = PROD<Int_t>()(prod, i + 1);
      }
      std::cout << std::endl;
      //       prInt_tf("MPI Result     = %d\n", value);
      //       prInt_tf("Correct Result = %d\n", sum);

      // require values to compare if everything is ok
      TMatrixD req_mat(size, size);
      TVectorD req_vec(size);

      for (auto i = 0; i < size; i++) {
         req_vec[i] = 1 * size;
         for (auto j = 0; j < size; j++) req_mat[i][j] = 1.0 * size;
      }

      // assertions
      assert(value == sum);
      assert(recv_vec == req_vec);
      assert(recv_mat == req_mat);
      assert(prod_value == prod);
      assert(min_value == 0);
      assert(max_value == (size - 1));
      assert(c_value.Im() == cplx_r.Im());
      assert(c_value.Re() == cplx_r.Re());
   }
}

void reduce_test_array(Int_t root = 0, Int_t count = 2)
{
   auto rank = COMM_WORLD.GetRank();
   auto size = COMM_WORLD.GetSize();

   Int_t vars[count];
   /////////////////////////
   // testing custom object//
   /////////////////////////
   TMatrixD send_mat[count];
   TVectorD send_vec[count];
   for (auto k = 0; k < count; k++) {
      vars[k] = rank;
      for (auto i = 0; i < size; i++) {
         send_mat[k].ResizeTo(size, size);
         send_vec[k].ResizeTo(size);
         send_vec[k][i] = 1;
         for (auto j = 0; j < size; j++) send_mat[k][i][j] = 1.0;
      }
   }
   Int_t values[count];

   COMM_WORLD.Reduce(vars, values, count, SUM, root); // testing custom object
   COMM_WORLD.Barrier();

   TMatrixD recv_mat[count];
   COMM_WORLD.Reduce(send_mat, recv_mat, count, SUM, root); // testing custom object
   COMM_WORLD.Barrier();

   TVectorD recv_vec[count];
   COMM_WORLD.Reduce(send_vec, recv_vec, count, SUM, root); // testing custom object
   COMM_WORLD.Barrier();

   if (rank == root) {
      Int_t sum = 0;
      for (Int_t i = 0; i < size; i++) {
         sum = SUM<Int_t>()(sum, i);
      }

      // require values to compare if everything is ok
      TMatrixD req_mat(size, size);
      TVectorD req_vec(size);
      for (auto i = 0; i < size; i++) {
         req_vec[i] = 1 * size;
         for (auto j = 0; j < size; j++) req_mat[i][j] = 1.0 * size;
      }

      std::cout << std::endl;
      for (auto i = 0; i < count; i++) {
         // assertions
         assert(values[i] == sum);
         assert(recv_mat[i] == req_mat);
         assert(recv_vec[i] == req_vec);
      }
   }
}

void reduce(Bool_t stressTest = kTRUE)
{
   TEnvironment env;
   if (COMM_WORLD.GetSize() == 1) return; // needed at least 2 process
   if (!stressTest) {
      reduce_test_scalar(0);
      reduce_test_array(0);
   } else {
      // stressTest
      for (auto i = 0; i < COMM_WORLD.GetSize(); i++) {
         reduce_test_scalar(i);
         reduce_test_array(i, COMM_WORLD.GetSize() * 2);
      }
   }
}
