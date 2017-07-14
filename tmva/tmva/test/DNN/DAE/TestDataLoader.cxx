// @(#)root/tmva/tmva/dnn:$Id$
// Author: Akshay Vashistha (ajatgd)

/*************************************************************************
 * Copyright (C) 2016, ajatgd                                            *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////
// Test the reference data loader implementation. //
////////////////////////////////////////////////////

#include "TestDataLoader.h"
#include "TMVA/DNN/Architectures/Reference.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::DAE;

int main() {
  testDataLoader<TReference<Double_t>>();
  std::cout << "Testing reference data loader: Mex. rel. error = ";
  std::cout << std::endl;
}
