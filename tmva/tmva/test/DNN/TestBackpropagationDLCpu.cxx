// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Concrete instantiation of the generic backpropagation test for //
// multi-threaded CPU architectures.                              //
////////////////////////////////////////////////////////////////////

#include "TMatrix.h"
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestBackpropagationDL.h"
#include "TRandom.h"

using namespace TMVA::DNN;

int main()
{
   using Scalar_t = Double_t;
   std::cout << "Testing Backpropagation:" << std::endl;

   double error;
   int iret = 0;

   gRandom->SetSeed(0);

   error = testBackpropagationWeightsLinear<TCpu<Scalar_t>>(0.1);
   if (error > 1e-3)
       iret++;

   error = testBackpropagationL1Regularization<TCpu<Scalar_t>>(1e-5); // relu can use large dx
   if (error > 1e-3)
       iret++;

   error = testBackpropagationL2Regularization<TCpu<Scalar_t>>(1.E-5);
   if (error > 1e-3)
       iret++;

   error = testBackpropagationBiasesLinear<TCpu<Scalar_t>>(0.1);  // use large dx since network is linear
   if (error > 1e-3)
       iret++;

   return iret;
}
