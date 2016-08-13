// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 19/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

 ///////////////////////////////////////////////////////////////////
 // Implementation of the activation functions for multi-threaded //
 // CPU architectures using tbb and BLAS.                         //
 ///////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cpu.h"
#include <math.h>
namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::IdentityDerivative(TCpuMatrix<Real_t> & B,
                                                   const TCpuMatrix<Real_t> &A)
{
   auto f = [](Real_t) {return 1.0;};
   B.Map(f);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::Relu(TCpuMatrix<Real_t> & B)
{
   auto f = [](Real_t x) {return (x < 0.0) ? 0.0 : x;};
   B.Map(f);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::ReluDerivative(TCpuMatrix<Real_t> & B,
                                               const TCpuMatrix<Real_t> &A)
{
   auto f = [](Real_t x) {return (x < 0.0) ? 0.0 : 1.0;};
   B.MapFrom(f, A);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::Sigmoid(TCpuMatrix<Real_t> & B)
{
   auto f = [](Real_t x) {return 1.0 / (1.0 + exp(-x));};
   B.Map(f);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::SigmoidDerivative(TCpuMatrix<Real_t> & B,
                                                  const TCpuMatrix<Real_t> &A)
{
   auto f = [](Real_t x) {
      Real_t sig = 1.0 / (1.0 + exp(-x));
      return sig * (1.0 - sig);
   };
   B.MapFrom(f, A);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::Tanh(TCpuMatrix<Real_t> & B)
{
   auto f = [](Real_t x) {return tanh(x);};
   B.Map(f);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::TanhDerivative(TCpuMatrix<Real_t> & B,
                                               const TCpuMatrix<Real_t> &A)
{
   auto f = [](Real_t x) {
      Real_t t = tanh(x);
      return 1 - t * t;
   };
   B.MapFrom(f, A);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::SymmetricRelu(TCpuMatrix<Real_t> & B)
{
   auto f = [](Real_t x) {return fabs(x);};
   B.Map(f);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::SymmetricReluDerivative(TCpuMatrix<Real_t> & B,
                                                        const TCpuMatrix<Real_t> &A)
{
   auto f = [](Real_t x) {
      return (x < 0.0) ? -1.0 : 1.0;
   };
   B.MapFrom(f, A);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::SoftSign(TCpuMatrix<Real_t> & B)
{
   auto f = [](Real_t x) {return x / (1 + fabs(x));};
   B.Map(f);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::SoftSignDerivative(TCpuMatrix<Real_t> & B,
                                                   const TCpuMatrix<Real_t> &A)
{
   auto f = [](Real_t x) {
      x = 1.0 + fabs(x);
      x = 1.0 / (x * x);
      return x;
   };
   B.MapFrom(f, A);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::Gauss(TCpuMatrix<Real_t> & B)
{
   auto f = [](Real_t x) {return exp(- x * x);};
   B.Map(f);
}

//______________________________________________________________________________
template<typename Real_t, bool doProfiling>
void TCpu<Real_t, doProfiling>::GaussDerivative(TCpuMatrix<Real_t> & B,
                                                const TCpuMatrix<Real_t> &A)
{
   auto f = [](Real_t x) {return - 2.0 * x * exp(- x * x);};
   B.MapFrom(f, A);
}

} // namespace DNN
} // namespace TMVA
