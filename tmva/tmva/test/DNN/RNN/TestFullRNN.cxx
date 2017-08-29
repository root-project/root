// @(#)root/tmva $Id$
// Author: Saurav Shekhar 02/08/17

/*************************************************************************
 * Copyright (C) 2017, Saurav Shekhar                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
//Testing RNNLayer for incrementing a number                      //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"
#include "TestFullRNN.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;


int main() {
   std::cout << "Training RNN to identity first";

   //testFullRNN(size_t batchSize, size_t stateSize, size_t inputSize, size_t outputSize)
   // reconstruct 8 bit vector
   // batchsize, statesize, inputsize, outputsize
   testFullRNN<TReference<double>>(2, 3, 2, 2) ;
   //testFullRNN<TReference<double>>(64, 10, 8, 8) ;
   //testFullRNN<TReference<double>>(3, 8, 100, 50) ;

   return 0;
}
