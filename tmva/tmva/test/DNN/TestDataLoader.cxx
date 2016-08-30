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
   Double_t error = testIdentity<TReference<Double_t>>();
   std::cout << "Testing reference data loader: Mex. rel. error = " << error;
   std::cout << std::endl;
}
