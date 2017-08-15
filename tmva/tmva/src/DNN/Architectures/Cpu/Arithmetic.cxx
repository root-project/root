// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 20/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////
//  Implementation of Helper arithmetic functions for the //
// multi-threaded CPU implementation of DNNs.             //
////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cpu.h"
#include "TMVA/DNN/Architectures/Cpu/Blas.h"
#include "tbb/tbb.h"

namespace TMVA
{
namespace DNN
{

//____________________________________________________________________________
template<typename Real_t>
void TCpu<Real_t>::Multiply(TCpuMatrix<Real_t> &C,
                            const TCpuMatrix<Real_t> &A,
                            const TCpuMatrix<Real_t> &B)
{
    int m = (int) A.GetNrows();
    int k = (int) A.GetNcols();
    int n = (int) B.GetNcols();

    char transa = 'N';
    char transb = 'N';

    Real_t alpha = 1.0;
    Real_t beta  = 0.0;

    const Real_t * APointer = A.GetRawDataPointer();
    const Real_t * BPointer = B.GetRawDataPointer();
          Real_t * CPointer = C.GetRawDataPointer();

    ::TMVA::DNN::Blas::Gemm(&transa, &transb, &m, &n, &k, &alpha,
                            APointer, &m, BPointer, &k, &beta, CPointer, &m);
}

//____________________________________________________________________________
template<typename Real_t>
void TCpu<Real_t>::TransposeMultiply(TCpuMatrix<Real_t> &C,
                                     const TCpuMatrix<Real_t> &A,
                                     const TCpuMatrix<Real_t> &B)
{
    int m = (int) A.GetNcols();
    int k = (int) A.GetNrows();
    int n = (int) B.GetNcols();

    char transa = 'T';
    char transb = 'N';

    Real_t alpha = 1.0;
    Real_t beta  = 0.0;

    const Real_t *APointer = A.GetRawDataPointer();
    const Real_t *BPointer = B.GetRawDataPointer();
          Real_t *CPointer = C.GetRawDataPointer();

    ::TMVA::DNN::Blas::Gemm(&transa, &transb, &m, &n, &k, &alpha,
                            APointer, &k, BPointer, &k, &beta, CPointer, &m);
}

//____________________________________________________________________________
template<typename Real_t>
void TCpu<Real_t>::Hadamard(TCpuMatrix<Real_t> &B,
                            const TCpuMatrix<Real_t> &A)
{
   const Real_t *dataA      = A.GetRawDataPointer();
         Real_t *dataB      = B.GetRawDataPointer();

   auto f = [&dataA, &dataB](UInt_t workerID)
   {
      dataB[workerID] *= dataA[workerID];
      return 0;
   };

   B.GetThreadExecutor().Map(f, ROOT::TSeqI(B.GetNElements()));
}

//____________________________________________________________________________
template<typename Real_t>
void TCpu<Real_t>::SumColumns(TCpuMatrix<Real_t> &B,
                                           const TCpuMatrix<Real_t> &A)
{
   int m = (int) A.GetNrows();
   int n = (int) A.GetNcols();
   int inc = 1;

   Real_t alpha = 1.0;
   Real_t beta  = 0.0;
   char   trans   = 'T';

   const Real_t * APointer = A.GetRawDataPointer();
         Real_t * BPointer = B.GetRawDataPointer();

   ::TMVA::DNN::Blas::Gemv(&trans, &m, &n, &alpha, APointer, &m,
                           TCpuMatrix<Real_t>::GetOnePointer(), &inc,
                           &beta, BPointer, &inc);
}

//____________________________________________________________________________
template<typename Real_t>
void TCpu<Real_t>::ScaleAdd(TCpuMatrix<Real_t> &B,
                            const TCpuMatrix<Real_t> &A,
                            Real_t alpha)
{
   int n = (int) (A.GetNcols() * A.GetNrows());
   int inc = 1;

   const Real_t *x = A.GetRawDataPointer();
   Real_t *y = B.GetRawDataPointer();

   ::TMVA::DNN::Blas::Axpy(&n, &alpha, x, &inc, y, &inc);
}

//____________________________________________________________________________
template<typename Real_t>
void TCpu<Real_t>::Copy(TCpuMatrix<Real_t> &B,
                        const TCpuMatrix<Real_t> &A)
{
   auto f = [](Real_t x) {return x;};
   B.MapFrom(f, A);
}


//____________________________________________________________________________
template<typename Real_t>
void TCpu<Real_t>::ScaleAdd(std::vector<TCpuMatrix<Real_t>> &B,
                            const std::vector<TCpuMatrix<Real_t>> &A,
                            Real_t alpha)
{
   for (size_t i = 0; i < B.size(); ++i) {
      ScaleAdd(B[i], A[i], alpha);
   }
}

//____________________________________________________________________________
template<typename Real_t>
void TCpu<Real_t>::Copy(std::vector<TCpuMatrix<Real_t>> &B,
                            const std::vector<TCpuMatrix<Real_t>> &A)
{
   for (size_t i = 0; i < B.size(); ++i) {
      Copy(B[i], A[i]);
   }
}


} // DNN
} // TMVA
