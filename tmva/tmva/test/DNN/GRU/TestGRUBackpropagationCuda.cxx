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
// Testing GRULayer backpropagation                               //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include <climits>
#include "TMVA/DNN/Architectures/TCudnn.h"
#include "TestGRUBackpropagation.h"
#include "TROOT.h"
#include "TMath.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;

int main() {
   std::cout << "Testing GRU backward pass\n";


   using Scalar_t = Double_t;
   using Architecture_t = TCudnn<Scalar_t>;

   gRandom->SetSeed(12345);
   Architecture_t::SetRandomSeed(gRandom->Integer(TMath::Limits<UInt_t>::Max()));

   using Scalar_t = Double_t;

   // timesteps, batchsize, statesize, inputsize  { fixed input, with dense layer, with extra GRU }
   testGRUBackpropagation<Architecture_t>(2, 1, 2, 3, 1e-5,{true,false,false},true );


   testGRUBackpropagation<Architecture_t>(1, 2, 1, 10, 1e-5);

   testGRUBackpropagation<Architecture_t>(1, 2, 3, 2, 1e-5);

   testGRUBackpropagation<Architecture_t>(2, 3, 4, 5, 1e-5);

   testGRUBackpropagation<Architecture_t>(4, 2, 10, 5, 1e-5);

   testGRUBackpropagation<Architecture_t>(5, 8, 10, 5, 1e-5);

   // with a dense layer
   testGRUBackpropagation<Architecture_t>(4, 4, 10, 20, 1e-5, {false, true});
   // with an additional GRU layer
   testGRUBackpropagation<Architecture_t>(4, 8, 10, 5, 1e-5, {false, true, true});

   // using a fixed input
   testGRUBackpropagation<Architecture_t>(3, 1, 10, 5, 1e-5, {true});

   return 0;
}
