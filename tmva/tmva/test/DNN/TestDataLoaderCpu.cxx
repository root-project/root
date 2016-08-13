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
#include "TestDataLoader.h"

using namespace TMVA::DNN;

int main ()
{
   std::cout << "Testing data loader:" << std::endl;
   Double_t error = testIdentity<TCpu<Double_t, false>>();
   std::cout << "Identity: " << error << std::endl;
   error = testSum<TCpu<Double_t, false>>();
   std::cout << "Sum     : " << error << std::endl;

   return 0;
}
