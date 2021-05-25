// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////
// Use the generic tests defined in TestMinimization.h to test the //
// training of Neural Networks for CUDA architectures.             //
/////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Minimizers.h"
#include "TestMinimization.h"

using namespace TMVA::DNN;

template<typename AFloat>
int test(double tol) {

   // create a dummy tensor to init a cuda/cudnn handle
   TCudaTensor<Float_t> dummy(1, 1);

   Double_t error = testMinimization<TCuda<Float_t>>();
   std::cout << "Gradient Descent: Maximum relative error = " << error << std::endl;
   if (error > tol) {
       return 1;
   }

   error = testMinimizationMomentum<TCuda<Float_t>>();
   std::cout << "Momentum:         Maximum relative error = " << error << std::endl;
    if (error > tol) {
       return 1;
   }

   error = testMinimizationWeights<TCuda<Float_t>>();
   std::cout << "Weighted Data:    Maximum relative error = " << error << std::endl;
   if (error > tol) {
      return 1;
   }
   return 0;
}
int main()
{
   bool fail = false;
   std::cout << "Testing minimization: (single precision)" << std::endl;
   fail |= test<Float_t>( 1.E-3);

   std::cout << std::endl << "Testing minimization: (double precision)" << std::endl;
   fail |= test<Double_t>(1.E-3);
   return fail;
}
