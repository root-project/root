// @(#)root/tmva $Id$
// Author: Saurav Shekhar 16/02/17

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
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestRecurrentBackpropagation.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;

int main() {
   std::cout << "Testing RNN backward pass\n";
   using Scalar_t = Double_t;

   // timesteps, batchsize, statesize, inputsize
   testRecurrentBackpropagation<TCpu<Scalar_t>>(1, 2, 1, 2, 1e-5);

   testRecurrentBackpropagation<TCpu<Scalar_t>>(1, 2, 3, 2, 1e-5); 

   testRecurrentBackpropagation<TCpu<Scalar_t>>(2, 3, 4, 5, 1e-5);


   // using a fixed input 
   testRecurrentBackpropagation<TCpu<Scalar_t>>(3, 1, 4, 5, 1e-10, false);


   return 0;
}
