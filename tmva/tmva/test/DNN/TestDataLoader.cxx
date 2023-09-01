// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 12/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////
// Test the reference data loader implementation. //
////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Reference.h"
#include "TestDataLoader.h"

using namespace TMVA::DNN;

int main ()
{
   using Scalar_t = Real_t;

   std::cout << "Testing data loader:" << std::endl;

   Scalar_t maximumError = 0.0;

   Scalar_t error = testSum<TReference<Scalar_t>>();
   std::cout << "Sum:      Maximum relative error = " << error << std::endl;
   maximumError = std::max(error, maximumError);
   error = testIdentity<TReference<Scalar_t>>();
   std::cout << "Identity: Maximum relative error = " << error << std::endl;
   maximumError = std::max(error, maximumError);

   if (maximumError > 1e-3) {
      return 1;
   }
}
