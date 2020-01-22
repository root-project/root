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
#include "TROOT.h"
#include "TMath.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;

int main() {

   bool debug = true;
   std::cout << "Testing RNN backward pass\n";

   ROOT::EnableImplicitMT(1);

   using Scalar_t = Double_t;

   gRandom->SetSeed(12345);
   TCpu<double>::SetRandomSeed(gRandom->Integer(TMath::Limits<UInt_t>::Max()));

   bool fail = false;
   if (debug) {
      //fail |=   testRecurrentBackpropagation<TCpu<Scalar_t>>(2, 1, 1, 2, 1e-5, {true, false, false}, true);
      //fail |= testRecurrentBackpropagation<TCpu<Scalar_t>>(1, 1, 1, 1, 1e-5, {true, true, false}, true);
      //fail |= testRecurrentBackpropagation<TCpu<Scalar_t>>(2, 3, 4, 5, 1e-5, {false, true, false}, true);
      fail |= testRecurrentBackpropagation<TCpu<Scalar_t>>(2, 1, 4, 5, 1e-5, {true, true, false}, true);
      return fail;
   }

   // timesteps, batchsize, statesize, inputsize  { fixed input, with dense layer, with extra RNN }

   fail |= testRecurrentBackpropagation<TCpu<Scalar_t>>(1, 2, 1, 2, 1e-5);

   fail |= testRecurrentBackpropagation<TCpu<Scalar_t>>(1, 2, 3, 2, 1e-5);

   fail |= testRecurrentBackpropagation<TCpu<Scalar_t>>(2, 3, 4, 5, 1e-5);

   fail |= testRecurrentBackpropagation<TCpu<Scalar_t>>(4, 2, 10, 5, 1e-5);
//   testRecurrentBackpropagation<TCpu<Scalar_t>>(4, 2, 5, 4, 1e-5, true, false, true);

   fail |= testRecurrentBackpropagation<TCpu<Scalar_t>>(5, 64, 10, 5, 1e-5);


   // using a fixed input
   fail |= testRecurrentBackpropagation<TCpu<Scalar_t>>(3, 1, 10, 5, 1e-5, {true});

   // with a dense layer
   fail |= testRecurrentBackpropagation<TCpu<Scalar_t>>(4, 32, 10, 20, 1e-5, {false, true});

   // with an additional RNN layer
   fail |= testRecurrentBackpropagation<TCpu<Scalar_t>>(4, 32, 10, 5, 1e-5, {false, true, true});


   return fail;
}
