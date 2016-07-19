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
#include "TMVA/DNN/Architectures/Cuda/Kernels.h"
#include "TMVA/DNN/Architectures/Cuda/Device.h"

namespace TMVA
{
namespace DNN
{

//____________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::Multiply(TCudaMatrix &C,
                                  const TCudaMatrix &A,
                                  const TCudaMatrix &B)
{
   int m, n, k;
   m = A.GetNrows();
   k = A.GetNcols();
   n = B.GetNcols();
   CudaDouble_t alpha = 1.0, beta = 0.0;

   // Compute C = beta * C + alpha * (A * B)
   cublasDgemm(A.GetCublasHandle(),
               CUBLAS_OP_N, CUBLAS_OP_N,
               m, n, k, & alpha,
               A.GetDataPointer(), m,   // *A, lda
               B.GetDataPointer(), k,   // *B, ldb
               & beta,                  // beta
               C.GetDataPointer(), m);  // *C, ldc
}

//____________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::TransposeMultiply(TCudaMatrix & C,
                                           const TCudaMatrix & A,
                                           const TCudaMatrix & B)
{
   int m, n, k;
   k = A.GetNrows();
   m = A.GetNcols();
   n = B.GetNcols();
   CudaDouble_t alpha = 1.0, beta = 0.0;

   // Compute C = beta * C + alpha * (A^T * B)
   cublasDgemm(A.GetCublasHandle(),
               CUBLAS_OP_T, CUBLAS_OP_N,
               m, n, k, & alpha,
               A.GetDataPointer(), k,     // *A, lda
               B.GetDataPointer(), k,     // *B, ldb
               & beta,                    // beta
               C.GetDataPointer(), m);    // *C, ldc
}

//____________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::Hadamard(TCudaMatrix &B,
                                  const TCudaMatrix &A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(B);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::Hadamard<<<gridDims, blockDims, 0, s>>>(B.GetDataPointer(),
                                                              A.GetDataPointer(),
                                                              A.GetNrows(),
                                                              A.GetNcols());
}

//____________________________________________________________________________
template<bool doProfiling>
CudaDouble_t TCuda<doProfiling>::Sum(const TCudaMatrix &A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);
   cudaStream_t s = A.GetComputeStream();

   TCudaMatrix::ResetDeviceReturn();
   ::TMVA::DNN::Cuda::ReduceMatrix<<<gridDims, blockDims, 0, s>>>(
       TCudaMatrix::GetDeviceReturnPointer(),
       A.GetDataPointer(),
       A.GetNrows(),
       A.GetNcols());
   return TCudaMatrix::GetDeviceReturn();
}

//____________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::SumColumns(TCudaMatrix &B, const TCudaMatrix &A)
{
   dim3 blockDims = TDevice::BlockDims();
   dim3 gridDims  = TDevice::GridDims(A);

   cudaMemset(B.GetDataPointer(), 0, A.GetNcols() * sizeof(CudaDouble_t));
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::SumColumns<<<gridDims, blockDims, 0, s>>>(B.GetDataPointer(),
                                                                A.GetDataPointer(),
                                                                A.GetNrows(),
                                                                A.GetNcols());
}

//____________________________________________________________________________
template<bool doProfiling>
void TCuda<doProfiling>::ScaleAdd(TCudaMatrix &B,
                                  const TCudaMatrix &A,
                                  CudaDouble_t alpha)
{
   cublasDaxpy(A.GetCublasHandle(), A.GetNoElements(), &alpha,
               A.GetDataPointer(), 1,
               B.GetDataPointer(), 1);
}

} // DNN
} // TMVA
