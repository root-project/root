// @(#)root/tmva $Id$
// Author: Saurav Shekhar 30/11/17

/*************************************************************************
 * Copyright (C) 2017, Saurav Shekhar                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Testing RNNLayer backpropagation                               //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"
#include "TestRecurrentBackpropagation.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;

int main() {
   std::cout << "Testing RNN backward pass\n";

   // timesteps, batchsize, statesize, inputsize
   testRecurrentBackpropagationWeights<TReference<double>>(1, 2, 3, 2, 1.0);

   testRecurrentBackpropagationBiases<TReference<double>>(1, 2, 3, 2, 1.0); 

   testRecurrentBackpropagationWeights<TReference<double>>(2, 3, 4, 5, 1.0);

   return 0;
}
