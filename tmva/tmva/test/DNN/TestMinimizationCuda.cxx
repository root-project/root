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

int main()
{
   std::cout << "Testing minimization: (single precision)" << std::endl;

   Double_t error = testMinimization<TCuda<Real_t>>();
   std::cout << "Gradient Descent: Maximum relative error = " << error << std::endl;
   if (error > 1) {
       return 1;
   }

   error = testMinimizationMomentum<TCuda<Real_t>>();
   std::cout << "Momentum:         Maximum relative error = " << error << std::endl;
    if (error > 1) {
       return 1;
   }

   error = testMinimizationWeights<TCuda<Real_t>>();
   std::cout << "Weighted Data:    Maximum relative error = " << error << std::endl;
   if (error > 1e-3) {
      return 1;
   }

   std::cout << std::endl << "Testing minimization: (double precision)" << std::endl;

   error = testMinimization<TCuda<Double_t>>();
   std::cout << "Gradient Descent: Maximum relative error = " << error << std::endl;
   if (error > 1e-3) {
       return 1;
   }

   error = testMinimizationMomentum<TCuda<Double_t>>();
   std::cout << "Momentum:         Maximum relative error = " << error << std::endl;
   if (error > 1e-3) {
       return 1;
   }

   error = testMinimizationWeights<TCuda<Double_t>>();
   std::cout << "Weighted Data:    Maximum relative error = " << error << std::endl;
   if (error > 1e-3) {
      return 1;
   }
}
