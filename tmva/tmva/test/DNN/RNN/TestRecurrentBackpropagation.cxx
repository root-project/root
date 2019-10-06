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

   testRecurrentBackpropagation<TReference<double>>(2, 1, 1, 2, 1e-5, {true, false, false}, true);

   return 0;

   // timesteps, batchsize, statesize, inputsize
   testRecurrentBackpropagation<TReference<double>>(1, 2, 1, 10, 1e-5);


   testRecurrentBackpropagation<TReference<double>>(2, 2, 1, 10, 1e-5);

   testRecurrentBackpropagation<TReference<double>>(1, 2, 2, 10, 1e-5);


   testRecurrentBackpropagation<TReference<double>>(2, 1, 2, 5, 1e-5);

   testRecurrentBackpropagation<TReference<double>>(4, 2, 3, 10, 1e-5);
   // using a fixed input
   testRecurrentBackpropagation<TReference<double>>(3, 1, 4, 5, 1e-5, {true});

   // with a dense layer
   testRecurrentBackpropagation<TReference<double>>(4, 32, 10, 5, 1e-5, {false, true});
   // with an additional RNN layer
   testRecurrentBackpropagation<TReference<double>>(4, 32, 10, 5, 1e-5, {false, true, true});

   return 0;
}
