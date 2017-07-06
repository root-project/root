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
// Train the multi-threaded CPU implementation of DNNs on a random //
// linear mapping. In the linear case the minimization problem is  //
// convex and the gradient descent training should converge to the //
// global minimum.                                                 //
/////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestMinimization.h"

using namespace TMVA::DNN;

int main()
{

   std::cout << "Testing minimization: (single precision)" << std::endl;

   Double_t error = testMinimization<TCpu<Real_t>>();
   std::cout << "Gradient Descent: Maximum relative error = " << error << std::endl;
   if (error > 1e-3) {
       return 1;
   }

   error = testMinimizationMomentum<TCpu<Real_t>>();
   std::cout << "Momentum:         Maximum relative error = " << error << std::endl;
   if (error > 1e-3) {
       return 1;
   }

   error = testMinimizationWeights<TCpu<Real_t>>();
   std::cout << "Weighted Data:    Maximum relative error = " << error << std::endl;
   if (error > 1e-3) {
      return 1;
   }

   std::cout << std::endl << "Testing minimization: (double precision)" << std::endl;

   error = testMinimization<TCpu<Double_t>>();
   std::cout << "Gradient Descent: Maximum relative error = " << error << std::endl;
   if (error > 1e-5) {
       return 1;
   }

   error = testMinimizationMomentum<TCpu<Double_t>>();
   std::cout << "Momentum:         Maximum relative error = " << error << std::endl;
   if (error > 1e-5) {
       return 1;
   }

   error = testMinimizationWeights<TCpu<Double_t>>();
   std::cout << "Weighted Data:    Maximum relative error = " << error << std::endl;
   if (error > 1e-3) {
      return 1;
   }

   return 0;
}
