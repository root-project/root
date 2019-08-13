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

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wshadow"

#include "tbb/tbb.h"

#pragma GCC diagnostic pop

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

    R__ASSERT((int) C.GetNrows() == m);
    R__ASSERT((int) C.GetNcols() == n);
    R__ASSERT((int) B.GetNrows() == k); 
    
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
                                     const TCpuMatrix<Real_t> &B,
                                     Real_t alpha, Real_t beta)
{
    int m = (int) A.GetNcols();
    int k = (int) A.GetNrows();
    int n = (int) B.GetNcols();

    R__ASSERT((int) C.GetNrows() == m);
    R__ASSERT((int) C.GetNcols() == n);
    R__ASSERT((int) B.GetNrows() == k); 
    
    char transa = 'T';
    char transb = 'N';

    //Real_t alpha = 1.0;
    //Real_t beta  = 0.0;

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

   size_t nElements =  A.GetNoElements();
   R__ASSERT(B.GetNoElements() == nElements); 
   size_t nSteps = TCpuMatrix<Real_t>::GetNWorkItems(nElements);

   auto f = [&](UInt_t workerID)
   {
      for (size_t j = 0; j < nSteps; ++j) {
         size_t idx = workerID+j;
         if (idx >= nElements) break; 
         dataB[idx] *= dataA[idx];
      }
      return 0;
   };

   if (nSteps < nElements) { 
#ifdef DL_USE_MTE
      B.GetThreadExecutor().Foreach(f, ROOT::TSeqI(0,nElements,nSteps));
#else
      for (size_t i = 0;  i < nElements ; i+= nSteps)
         f(i);
#endif
   }
   else {
      f(0); 
   }
}

//____________________________________________________________________________
template<typename Real_t>
void TCpu<Real_t>::Hadamard(TCpuTensor<Real_t> &B,
                            const TCpuTensor<Real_t> &A)
{
   const Real_t *dataA      = A.GetRawDataPointer();
   Real_t *dataB      = B.GetRawDataPointer();

   size_t nElements =  A.GetNoElements();
   R__ASSERT(B.GetNoElements() == nElements); 
   size_t nSteps = TCpuMatrix<Real_t>::GetNWorkItems(nElements);

   auto f = [&](UInt_t workerID)
   {
      for (size_t j = 0; j < nSteps; ++j) {
         size_t idx = workerID+j;
         if (idx >= nElements) break; 
         dataB[idx] *= dataA[idx];
      }
      return 0;
   };

   if (nSteps < nElements) { 
#ifdef DL_USE_MTE
      TMVA::Config::Instance().GetThreadExecutor().Foreach(f, ROOT::TSeqI(0,nElements,nSteps));
#else
      for (size_t i = 0;  i < nElements ; i+= nSteps)
         f(i);
#endif
   }
   else {
      f(0); 
   }
}

////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
/// \brief Checks two matrices for element-wise equality.
/// \tparam Real_t An architecture-specific floating point number type.
/// \param A The first matrix.
/// \param B The second matrix.
/// \param epsilon Equality tolerance, needed to address floating point arithmetic.
/// \return Whether the two matrices can be considered equal element-wise
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template<typename Real_t>
bool TCpu<Real_t>::AlmostEquals(const TCpuMatrix<Real_t> &A, const TCpuMatrix<Real_t> &B, double epsilon)
{
    if (A.GetNrows() != B.GetNrows() || A.GetNcols() != B.GetNcols()) {
        Fatal("AlmostEquals", "The passed matrices have unequal shapes.");
    }

    const Real_t *dataA = A.GetRawDataPointer();
    const Real_t *dataB = B.GetRawDataPointer();
    size_t nElements =  A.GetNoElements();

    for(size_t i = 0; i < nElements; i++) {
        if(fabs(dataA[i] - dataB[i]) > epsilon) return false;
    }
    return true;
}

//____________________________________________________________________________
template<typename Real_t>
void TCpu<Real_t>::SumColumns(TCpuMatrix<Real_t> &B,
                              const TCpuMatrix<Real_t> &A,
                              Real_t alpha, Real_t beta)
{
   int m = (int) A.GetNrows();
   int n = (int) A.GetNcols();
   int inc = 1;

   // Real_t alpha = 1.0;
   //Real_t beta  = 0.0;
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
void TCpu<Real_t>::ScaleAdd(TCpuTensor<Real_t> &B,
                            const TCpuTensor<Real_t> &A,
                            Real_t alpha)
{
   // should re-implemented at tensor level
   for (size_t i = 0; i < B.GetFirstSize(); ++i) {
      TCpuMatrix<Real_t> B_m = B.At(i).GetMatrix(); 
      ScaleAdd(B_m, A.At(i).GetMatrix(), alpha);
   }
}

//____________________________________________________________________________
template<typename Real_t>
void TCpu<Real_t>::Copy(TCpuTensor<Real_t> &B,
                            const TCpuTensor<Real_t> &A)
{

   auto f = [](Real_t x) {return x;};
   B.MapFrom(f, A);
}

//____________________________________________________________________________
template <typename Real_t>
void TCpu<Real_t>::ConstAdd(TCpuMatrix<Real_t> &A, Real_t beta)
{
   auto f = [beta](Real_t x) { return x + beta; };
   A.Map(f);
}

//____________________________________________________________________________
template <typename Real_t>
void TCpu<Real_t>::ConstMult(TCpuMatrix<Real_t> &A, Real_t beta)
{
   auto f = [beta](Real_t x) { return x * beta; };
   A.Map(f);
}

//____________________________________________________________________________
template <typename Real_t>
void TCpu<Real_t>::ReciprocalElementWise(TCpuMatrix<Real_t> &A)
{
   auto f = [](Real_t x) { return 1.0 / x; };
   A.Map(f);
}

//____________________________________________________________________________
template <typename Real_t>
void TCpu<Real_t>::SquareElementWise(TCpuMatrix<Real_t> &A)
{
   auto f = [](Real_t x) { return x * x; };
   A.Map(f);
}

//____________________________________________________________________________
template <typename Real_t>
void TCpu<Real_t>::SqrtElementWise(TCpuMatrix<Real_t> &A)
{
   auto f = [](Real_t x) { return sqrt(x); };
   A.Map(f);
}

/// Adam updates 
//____________________________________________________________________________
template<typename Real_t>
void TCpu<Real_t>::AdamUpdate(TCpuMatrix<Real_t> &A, const TCpuMatrix<Real_t> & M, const TCpuMatrix<Real_t> & V, Real_t alpha, Real_t eps)
{
   // ADAM update the weights.
   // Weight = Weight - alpha * M / (sqrt(V) + epsilon)
   Real_t * a = A.GetRawDataPointer();
   const Real_t * m = M.GetRawDataPointer(); 
   const Real_t * v = V.GetRawDataPointer();
   for (size_t index = 0; index < A.GetNoElements() ; ++index) {
      a[index] = a[index] - alpha * m[index]/( sqrt(v[index]) + eps);
   }
}

//____________________________________________________________________________
template<typename Real_t>
void TCpu<Real_t>::AdamUpdateFirstMom(TCpuMatrix<Real_t> &A, const TCpuMatrix<Real_t> & B, Real_t beta)
{
   // First momentum weight gradient update for ADAM
   // Mt = beta1 * Mt-1 + (1-beta1) * WeightGradients
   Real_t * a = A.GetRawDataPointer();
   const Real_t * b = B.GetRawDataPointer();
   for (size_t index = 0; index < A.GetNoElements() ; ++index) {
      a[index] = beta * a[index] + (1.-beta) * b[index];
   }
}
//____________________________________________________________________________
template<typename Real_t>
void TCpu<Real_t>::AdamUpdateSecondMom(TCpuMatrix<Real_t> &A, const TCpuMatrix<Real_t> & B, Real_t beta)
{
   // Second momentum weight gradient update for ADAM 
   // Vt = beta2 * Vt-1 + (1-beta2) * WeightGradients^2
   Real_t * a = A.GetRawDataPointer();
   const Real_t * b = B.GetRawDataPointer();
   for (size_t index = 0; index < A.GetNoElements() ; ++index) {
      a[index] = beta * a[index] + (1.-beta) * b[index] * b[index];
   }
}


} // DNN
} // TMVA
