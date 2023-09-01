// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////
// Test the Neural Network training using the reference   //
// implementation.                                        //
//                                                        //
// Calls the generic testMinimization function defined in //
// TestMinimization.cpp for the reference architecture.   //
////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"
#include "TestMinimization.h"

using namespace TMVA::DNN;

int main()
{
   std::cout << "Testing minimization: (single precision)" << std::endl;

   Double_t error = testMinimization<TReference<Real_t>>();
   std::cout << "Gradient Descent: Maximum relative error = " << error << std::endl;
   if (error > 1e-3) {
      return 1;
   }

   error = testMinimizationMomentum<TReference<Real_t>>();
   std::cout << "Momentum:         Maximum relative error = " << error << std::endl;
   if (error > 1e-3) {
      return 1;
   }

   error = testMinimizationWeights<TReference<Real_t>>();
   std::cout << "Weighted Data:    Maximum relative error = " << error << std::endl;
   if (error > 1e-3) {
      return 1;
   }

   std::cout << std::endl << "Testing minimization: (double precision)" << std::endl;

   error = testMinimization<TReference<Double_t>>();
   std::cout << "Gradient Descent: Maximum relative error = " << error << std::endl;
   if (error > 1e-5) {
      return 1;
   }

   error = testMinimizationMomentum<TReference<Double_t>>();
   std::cout << "Momentum:         Maximum relative error = " << error << std::endl;
   if (error > 1e-5) {
      return 1;
   }

   error = testMinimizationWeights<TReference<Double_t>>();
   std::cout << "Weighted Data:    Maximum relative error = " << error << std::endl;
   if (error > 1e-3) {
      return 1;
   }

   return 0;
}
