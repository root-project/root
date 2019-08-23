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
                                      const TCudaMatrix<float> & B,
                                      float alpha, float beta)
{
   int m, n, k;
   k = A.GetNrows();
   m = A.GetNcols();
   n = B.GetNcols();
   //float alpha = 1.0, beta = 0.0;

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
                                      const TCudaMatrix<double> & B,
                                      double alpha, double beta)
{
   int m, n, k;
   k = A.GetNrows();
   m = A.GetNcols();
   n = B.GetNcols();
   //double alpha = 1.0, beta = 0.0;

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
void TCuda<AFloat>::Hadamard(TCudaTensor<AFloat> & B,
                             const TCudaTensor<AFloat> &A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   int ncols = A.GetFirstSize();   // ncols (size X)
   int nrows = A.GetFirstStride(); // nrows  (y size)
   if (ncols == 1) { 
      ncols = A.GetWSize(); 
      nrows = A.GetHSize(); 
   }
   
   dim3 gridDims  = TDevice::GridDims2D(nrows, ncols);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::Hadamard<<<gridDims, blockDims, 0, s>>>(B.GetDataPointer(),
                                                              A.GetDataPointer(),
                                                              nrows,ncols);
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
                              const TCudaMatrix<float> & A,
                              float alpha, float beta)
{
   int m, n;
   m = A.GetNrows();
   n = A.GetNcols();
   //float alpha = 1.0, beta = 0.0;

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
                               const TCudaMatrix<double> & A,
                               double alpha, double beta)
{
   int m, n;
   m = A.GetNrows();
   n = A.GetNcols();
   //double alpha = 1.0, beta = 0.0;

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

template<>
void TCuda<float>::SumRows(TCudaMatrix<float> & B,
                           const TCudaMatrix<float> & A)
{
    int m, n;
    m = A.GetNrows();
    n = A.GetNcols();
    float alpha = 1.0, beta = 0.0;

    cudaStream_t s = A.GetComputeStream();
    cublasSetStream(A.GetCublasHandle(), s);

    // Compute C = beta * C + alpha * (A * B)
    cublasSgemv(A.GetCublasHandle(), CUBLAS_OP_N,
                m, n, & alpha,
                A.GetDataPointer(), m,             // *A, lda
                TCudaMatrix<float>::GetOnes(), 1, // *x, incx
                & beta, B.GetDataPointer(), 1);    // beta, *y, incy

    B.SetComputeStream(s);
}

//____________________________________________________________________________
template<>
void TCuda<double>::SumRows(TCudaMatrix<double> & B,
                            const TCudaMatrix<double> & A)
{
    int m, n;
    m = A.GetNrows();
    n = A.GetNcols();
    double alpha = 1.0, beta = 0.0;

    cudaStream_t s = A.GetComputeStream();
    cublasSetStream(A.GetCublasHandle(), s);

    // Compute C = beta * C + alpha * (A * B)
    cublasDgemv(A.GetCublasHandle(), CUBLAS_OP_N,
                m, n, & alpha,
                A.GetDataPointer(), m,             // *A, lda
                TCudaMatrix<double>::GetOnes(), 1, // *x, incx
                & beta, B.GetDataPointer(), 1);    // beta, *y, incy

    B.SetComputeStream(s);
}



////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Checks two matrices for element-wise equality.
/// \tparam AFloat An architecture-specific floating point number type.
/// \param A The first matrix.
/// \param B The second matrix.
/// \param epsilon Equality tolerance, needed to address floating point arithmetic.
/// \return Whether the two matrices can be considered equal element-wise
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename AFloat>
bool TCuda<AFloat>::AlmostEquals(const TCudaMatrix<AFloat> &A, const TCudaMatrix<AFloat> &B, double epsilon)
{
    if (A.GetNrows() != B.GetNrows() || A.GetNcols() != B.GetNcols()) {
        Fatal("AlmostEquals", "The passed matrices have unequal shapes.");
    }

    dim3 blockDims = TDevice::BlockDims2D();
    dim3 gridDims  = TDevice::GridDims2D(A);
    cudaStream_t s = A.GetComputeStream();

    bool * dResult = 0;
    cudaMalloc((void**) &dResult, sizeof(bool));
    cudaMemset(dResult, 1, sizeof(bool));

    ::TMVA::DNN::Cuda::AlmostEquals<<<gridDims, blockDims, 0, s>>>(dResult, A.GetDataPointer(), B.GetDataPointer(),
                                                                   epsilon, A.GetNrows(), A.GetNcols());

    bool result;
    cudaMemcpy(&result, dResult, sizeof(bool), cudaMemcpyDeviceToHost);
    cudaFree(dResult);

    return result;
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
template<typename AFloat>
void TCuda<AFloat>::ScaleAdd(TCudaTensor<AFloat> & B,
                            const TCudaTensor<AFloat> & A,
                            AFloat alpha)
{
   // should re-implemented at tensor level
   for (size_t i = 0; i < A.GetFirstSize(); ++i) {
      TCudaMatrix<AFloat> B_m = B.At(i).GetMatrix();
      TCudaMatrix<AFloat> A_m = A.At(i).GetMatrix();  
      ScaleAdd(B_m, A_m, alpha);
   }
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::ConstAdd(TCudaMatrix<AFloat> &A, AFloat beta)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::ConstAdd<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       beta,
       (int) A.GetNrows(),
       (int) A.GetNcols());
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::ConstMult(TCudaMatrix<AFloat> &A, AFloat beta)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::ConstMult<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       beta,
       (int) A.GetNrows(),
       (int) A.GetNcols());
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::ReciprocalElementWise(TCudaMatrix<AFloat> &A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::ReciprocalElementWise<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::SquareElementWise(TCudaMatrix<AFloat> &A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::SquareElementWise<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::SqrtElementWise(TCudaMatrix<AFloat> &A)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::SqrtElementWise<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols());
}

/// Adam updates 
//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::AdamUpdate(TCudaMatrix<AFloat> &A, const TCudaMatrix<AFloat> & M, const TCudaMatrix<AFloat> & V, AFloat alpha, AFloat eps)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::AdamUpdate<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       M.GetDataPointer(),
       V.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols(),
       alpha, eps);
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::AdamUpdateFirstMom(TCudaMatrix<AFloat> &A, const TCudaMatrix<AFloat> & B, AFloat beta)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::AdamUpdateFirstMom<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       B.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols(), beta);
}

//____________________________________________________________________________
template<typename AFloat>
void TCuda<AFloat>::AdamUpdateSecondMom(TCudaMatrix<AFloat> &A, const TCudaMatrix<AFloat> & B, AFloat beta)
{
   dim3 blockDims = TDevice::BlockDims2D();
   dim3 gridDims  = TDevice::GridDims2D(A);
   cudaStream_t s = A.GetComputeStream();
   ::TMVA::DNN::Cuda::AdamUpdateSecondMom<<<gridDims, blockDims, 0, s>>>(
       A.GetDataPointer(),
       B.GetDataPointer(),
       (int) A.GetNrows(),
       (int) A.GetNcols(), beta);
}







   
} // DNN
} // TMVA
