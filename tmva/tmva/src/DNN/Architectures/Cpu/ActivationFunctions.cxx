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
 // CPU architectures using Roots TThreadExecutor and BLAS.            //
 ///////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cpu.h"
#include <math.h>

#ifdef R__HAS_VDT
#include "vdt/tanh.h"
#define TANH_IMPL_X   vdt::fast_tanhf(x)
#else 
#define TANH_IMPL_X    tanh(x)
#endif


namespace TMVA
{
namespace DNN
{

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::IdentityDerivative(TCpuMatrix<AFloat> & B,
                                      const TCpuMatrix<AFloat> &/*A*/)
{
   auto f = [](AFloat) {return 1.0;};
   B.Map(f);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::Relu(TCpuMatrix<AFloat> & B)
{
   auto f = [](AFloat x) {return (x < 0.0) ? 0.0 : x;};
   B.Map(f);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::ReluDerivative(TCpuMatrix<AFloat> & B,
                                               const TCpuMatrix<AFloat> &A)
{
   auto f = [](AFloat x) {return (x < 0.0) ? 0.0 : 1.0;};
   B.MapFrom(f, A);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::Sigmoid(TCpuMatrix<AFloat> & B)
{
   auto f = [](AFloat x) {return 1.0 / (1.0 + exp(-x));};
   B.Map(f);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::SigmoidDerivative(TCpuMatrix<AFloat> & B,
                                     const TCpuMatrix<AFloat> &A)
{
   auto f = [](AFloat x) {
      AFloat sig = 1.0 / (1.0 + exp(-x));
      return sig * (1.0 - sig);
   };
   B.MapFrom(f, A);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::Tanh(TCpuMatrix<AFloat> & B)
{
   auto f = [](AFloat x) {return TANH_IMPL_X;};
   B.Map(f);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::TanhDerivative(TCpuMatrix<AFloat> & B,
                                  const TCpuMatrix<AFloat> &A)
{
   auto f = [](AFloat x) {
      AFloat t = TANH_IMPL_X;
      return 1 - t * t;
   };
   B.MapFrom(f, A);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::SymmetricRelu(TCpuMatrix<AFloat> & B)
{
   auto f = [](AFloat x) {return fabs(x);};
   B.Map(f);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::SymmetricReluDerivative(TCpuMatrix<AFloat> & B,
                                           const TCpuMatrix<AFloat> &A)
{
   auto f = [](AFloat x) {
      return (x < 0.0) ? -1.0 : 1.0;
   };
   B.MapFrom(f, A);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::SoftSign(TCpuMatrix<AFloat> & B)
{
   auto f = [](AFloat x) {return x / (1 + fabs(x));};
   B.Map(f);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::SoftSignDerivative(TCpuMatrix<AFloat> & B,
                                      const TCpuMatrix<AFloat> &A)
{
   auto f = [](AFloat x) {
      x = 1.0 + fabs(x);
      x = 1.0 / (x * x);
      return x;
   };
   B.MapFrom(f, A);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::Gauss(TCpuMatrix<AFloat> & B)
{
   auto f = [](AFloat x) {return exp(- x * x);};
   B.Map(f);
}

//______________________________________________________________________________
template<typename AFloat>
void TCpu<AFloat>::GaussDerivative(TCpuMatrix<AFloat> & B,
                                   const TCpuMatrix<AFloat> &A)
{
   auto f = [](AFloat x) {return - 2.0 * x * exp(- x * x);};
   B.MapFrom(f, A);
}

} // namespace DNN
} // namespace TMVA
