// @(#)root/tmva $Id$
// Author: Surya S Dwivedi 26/06/2019

/*************************************************************************
 * Copyright (C) 2019, Surya S Dwivedi                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Testing LSTMLayer backpropagation                               //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/TCudnn.h"
#include "TestLSTMBackpropagation.h"
#include "TROOT.h"
#include "TMath.h"


using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;

int main() {
   std::cout << "Testing LSTM backward pass\n";

   //ROOT::EnableImplicitMT(1);

   using Scalar_t = Double_t;
   using Architecture_t = TCudnn<Scalar_t>;

   gRandom->SetSeed(12345);
   Architecture_t::SetRandomSeed(gRandom->Integer(TMath::Limits<UInt_t>::Max()));

   bool debug = true;

   // timesteps, batchsize, statesize, inputsize  { fixed input, with dense layer, with extra LSTM }

   testLSTMBackpropagation<Architecture_t>(2, 1, 2, 3, 1e-5, {}, debug);

   gRandom->SetSeed(0);
   testLSTMBackpropagation<Architecture_t>(2, 2, 3, 4, 1e-5, {}, debug);


   testLSTMBackpropagation<Architecture_t>(2, 3, 4, 5, 1e-5);


   testLSTMBackpropagation<Architecture_t>(4, 2, 10, 5, 1e-5);

   // large batch size fail for numerical reason
   testLSTMBackpropagation<Architecture_t>(5, 8, 10, 5, 1e-5, {} );

   // with a dense layer
   testLSTMBackpropagation<Architecture_t>(4, 4, 10, 20, 1e-5, {false, true});
   // with an additional LSTM layer
   testLSTMBackpropagation<Architecture_t>(4, 8, 10, 5, 1e-5, {false, true, true});

   // using a fixed input (this fails )
   testLSTMBackpropagation<Architecture_t>(3, 1, 10, 6, 1e-5, {true}, debug);

   return 0;
}
