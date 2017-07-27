// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 13/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////
// Contains additional arithmetic functions required by the CUDA //
// neural network implementation.                                //
///////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"
#include "Kernels.cuh"

namespace TMVA
{
namespace DNN
{

//____________________________________________________________________________
template<>
void TCuda<float>::Multiply(TCudaMatrix<float> &C,
                             const TCudaMatrix<float> &A,
                             const TCudaMatrix<float> &B)
{
   int m, n, k;
   m = A.GetNrows();
   k = A.GetNcols();
   n = B.GetNcols();
   float alpha = 1.0, beta = 0.0;

   cudaStream_t s = A.GetComputeStream();
   cublasSetStream(A.GetCublasHandle(), s);

   // Compute C = beta * C + alpha * (A * B)
   cublasSgemm(A.GetCublasHandle(),
               CUBLAS_OP_N, CUBLAS_OP_N,
               m, n, k, & alpha,
               A.GetDataPointer(), m,   // *A, lda
               B.GetDataPointer(), k,   // *B, ldb
               & beta,                  // beta
               C.GetDataPointer(), m);  // *C, ldc

   C.SetComputeStream(s);
}

//____________________________________________________________________________
template<>
void TCuda<double>::Multiply(TCudaMatrix<double> &C,
                             const TCudaMatrix<double> &A,
                             const TCudaMatrix<double> &B)
{
   int m, n, k;
   m = A.GetNrows();
   k = A.GetNcols();
   n = B.GetNcols();
   double alpha = 1.0, beta = 0.0;

   cudaStream_t s = A.GetComputeStream();
   cublasSetStream(A.GetCublasHandle(), s);

   // Compute C = beta * C + alpha * (A * B)
   cublasDgemm(A.GetCublasHandle(),
               CUBLAS_OP_N, CUBLAS_OP_N,
               m, n, k, & alpha,
               A.GetDataPointer(), m,   // *A, lda
               B.GetDataPointer(), k,   // *B, ldb
               & beta,                  // beta
               C.GetDataPointer(), m);  // *C, ldc

   C.SetComputeStream(s);
}

//____________________________________________________________________________
template<>
void TCuda<float>::TransposeMultiply(TCudaMatrix<float> & C,
                                      const TCudaMatrix<float> & A,
                                      const TCudaMatrix<float> & B)
{
   int m, n, k;
   k = A.GetNrows();
   m = A.GetNcols();
   n = B.GetNcols();
   float alpha = 1.0, beta = 0.0;

   cudaStream_t s = A.GetComputeStream();
   cublasSetStream(A.GetCublasHandle(), s);

   // Compute C = beta * C + alpha * (A^T * B)
   cublasSgemm(A.GetCublasHandle(),
               CUBLAS_OP_T, CUBLAS_OP_N,
               m, n, k, & alpha,
               A.GetDataPointer(), k,     // *A, lda
               B.GetDataPointer(), k,     // *B, ldb
               & beta,                    // beta
               C.GetDataPointer(), m);    // *C, ldc

   C.SetComputeStream(s);
}
//____________________________________________________________________________
template<>
void TCuda<double>::TransposeMultiply(TCudaMatrix<double> & C,
                                      const TCudaMatrix<double> & A,
                                      const TCudaMatrix<double> & B)
{
   int m, n, k;
   k = A.GetNrows();
   m = A.GetNcols();
   n = B.GetNcols();
   double alpha = 1.0, beta = 0.0;

   cudaStream_t s = A.GetComputeStream();
   cublasSetStream(A.GetCublasHandle(), s);

   // Compute C = beta * C + alpha * (A^T * B)
   cublasDgemm(A.GetCublasHandle(),
               CUBLAS_OP_T, CUBLAS_OP_N,
               m, n, k, & alpha,
               A.GetDataPointer(), k,     // *A, lda
               B.GetDataPointer(), k,     // *B, ldb
               & beta,                    // beta
               C.GetDataPointer(), m);    // *C, ldc

   C.SetComputeStream(s);
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::Hadamard(TCudaMatrix<AFloat> & B,
                             const TCudaMatrix<AFloat> &A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(B);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::Hadamard<<<gridDims, blockDims, 0, s>>>(B.GetDataPointer(),
                                                              A.GetDataPointer(),
                                                              A.GetNrows(),
                                                              A.GetNcols());
   B.SetComputeStream(s);
}

//____________________________________________________________________________
template<typename AFloat>
AFloat TCuda<AFloat>::Sum(const TCudaMatrix<AFloat> & A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();

   TCudaMatrix<AFloat>::ResetDeviceReturn();
   ::TMVA::DNN::Cuda::ReduceMatrix<<<gridDims, blockDims, 0, s>>>(
       TCudaMatrix<AFloat>::GetDeviceReturnPointer(),
       A.GetDataPointer(),
       A.GetNrows(),
       A.GetNcols());
   return TCudaMatrix<AFloat>::GetDeviceReturn();
}

//____________________________________________________________________________
template<>
void TCuda<float>::SumColumns(TCudaMatrix<float> & B,
                               const TCudaMatrix<float> & A)
{
   int m, n;
   m = A.GetNrows();
   n = A.GetNcols();
   float alpha = 1.0, beta = 0.0;

   cudaStream_t s = A.GetComputeStream();
   cublasSetStream(A.GetCublasHandle(), s);

   // Compute C = beta * C + alpha * (A * B)
   cublasSgemv(A.GetCublasHandle(), CUBLAS_OP_T,
               m, n, & alpha,
               A.GetDataPointer(), m,             // *A, lda
               TCudaMatrix<float>::GetOnes(), 1, // *x, incx
               & beta, B.GetDataPointer(), 1);    // beta, *y, incy

   B.SetComputeStream(s);
}

//____________________________________________________________________________
template<>
void TCuda<double>::SumColumns(TCudaMatrix<double> & B,
                               const TCudaMatrix<double> & A)
{
   int m, n;
   m = A.GetNrows();
   n = A.GetNcols();
   double alpha = 1.0, beta = 0.0;

   cudaStream_t s = A.GetComputeStream();
   cublasSetStream(A.GetCublasHandle(), s);

   // Compute C = beta * C + alpha * (A * B)
   cublasDgemv(A.GetCublasHandle(), CUBLAS_OP_T,
               m, n, & alpha,
               A.GetDataPointer(), m,             // *A, lda
               TCudaMatrix<double>::GetOnes(), 1, // *x, incx
               & beta, B.GetDataPointer(), 1);    // beta, *y, incy

   B.SetComputeStream(s);
}

//____________________________________________________________________________
template<>
void TCuda<float>::ScaleAdd(TCudaMatrix<float> & B,
                            const TCudaMatrix<float> & A,
                            float alpha)
{
   cudaStream_t s = 0;
   cublasSetStream(A.GetCublasHandle(), s);
   cublasSaxpy(A.GetCublasHandle(), A.GetNoElements(), &alpha,
               A.GetDataPointer(), 1,
               B.GetDataPointer(), 1);
}

//____________________________________________________________________________
template<>
void TCuda<double>::ScaleAdd(TCudaMatrix<double> & B,
                             const TCudaMatrix<double> & A,
                             double alpha)
{
   cudaStream_t s = 0;
   cublasSetStream(A.GetCublasHandle(), s);
   cublasDaxpy(A.GetCublasHandle(), A.GetNoElements(), &alpha,
               A.GetDataPointer(), 1,
               B.GetDataPointer(), 1);
}

//____________________________________________________________________________
template<>
void TCuda<float>::ScaleAdd(std::vector<TCudaMatrix<float>> & B,
                            const std::vector<TCudaMatrix<float>> & A,
                            float alpha)
{
   for (size_t i = 0; i < A.size(); ++i) {
      ScaleAdd(B[i], A[i], alpha);
   }
}

//____________________________________________________________________________
template<>
void TCuda<double>::ScaleAdd(std::vector<TCudaMatrix<double>> & B,
                            const std::vector<TCudaMatrix<double>> & A,
                            double alpha)
{
   for (size_t i = 0; i < A.size(); ++i) {
      ScaleAdd(B[i], A[i], alpha);
   }
}

} // DNN
} // TMVA
