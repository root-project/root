// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 08/08/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////
// Test the generic data loader for the CUDA implementation. //
///////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cuda.h"
#include "TestDataLoader.h"

using namespace TMVA::DNN;

int main()
{
   std::cout << "Testing data loader:" << std::endl;
   using Scalar_t = Real_t;

   Scalar_t maximumError = 0.0;

   Scalar_t error = testSum<TCuda<Scalar_t>>();
   std::cout << "Sum:      Maximum relative error = " << error << std::endl;
   maximumError = std::max(error, maximumError);
   error = testIdentity<TCuda<Scalar_t>>();
   std::cout << "Identity: Maximum relative error = " << error << std::endl;
   maximumError = std::max(error, maximumError);

   if (maximumError > 1e-3) {
      return 1;
   }
   return 0;
}





