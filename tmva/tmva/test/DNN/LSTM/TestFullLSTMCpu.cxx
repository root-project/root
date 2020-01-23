// @(#)root/tmva $Id$
// Author: Surya S Dwivedi 02/07/19

/*************************************************************************
 * Copyright (C) 2019, Surya S Dwivedi                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Testing full LSTM network (for CPU)                            //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestFullLSTM.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;


int main() {
   std::cout << "Training LSTM to identity first";

   //testFullLSTM(size_t batchSize, size_t stateSize, size_t inputSize, size_t outputSize)
   // reconstruct 8 bit vector
   // batchsize, statesize, inputsize, outputsize
   testFullLSTM<TCpu<double>>(2, 3, 2, 2) ;
   //testFullLSTM<TReference<double>>(64, 10, 8, 8) ;
   //testFullLSTM<TReference<double>>(3, 8, 100, 50) ;

   // test a full LSTM with 5 time steps and different signal/backgrund time dependent shapes
   // batchsize, statesize , inputsize, seed
   int seed = 111;
   std::cout << "Training LSTM to simple time dependent data ";
   testFullLSTM2<TCpu<double>>(64, 10, 5, seed) ;

   return 0;
}
