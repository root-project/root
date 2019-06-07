// @(#)root/tmva $Id$
// Author: Harshit Prasad 27/05/18

/*************************************************************************
 * Copyright (C) 2018, Harshit Prasad                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Testing LSTM-Layer forward pass for Reference implementation   //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"
#include "TestLSTMForwardPass.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::LSTM;

int main() {

    std::cout << "Testing LSTM Forward pass:\n";

    // timesteps, batchsize, statesize, inputsize
    std::cout << testForwardPass<TReference<double>>(1, 2, 3, 2)  << "\n";
    std::cout << testForwardPass<TReference<double>>(1, 8, 100, 50)  << "\n";
    std::cout << testForwardPass<TReference<double>>(5, 9, 128, 64)  << "\n";

   return 0;
}
