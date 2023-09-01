// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 21/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////
// Test the multi-threaded CPU data loader implementation. //
/////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestTensorDataLoader.h"

using namespace TMVA::DNN;

int main ()
{
   using Scalar_t = Real_t;

   std::cout << "Testing data loader:" << std::endl;

   Scalar_t maximumError = 0.0;

   Scalar_t error = testSum<TCpu<Scalar_t>>();
   std::cout << "Sum:      Maximum relative error = " << error << std::endl;
   maximumError = std::max(error, maximumError);
   /*error = testIdentity<TCpu<Scalar_t>>();
   std::cout << "Identity: Maximum relative error = " << error << std::endl;
   maximumError = std::max(error, maximumError);*/

   if (maximumError > 1e-3) {
      return 1;
   }
}
