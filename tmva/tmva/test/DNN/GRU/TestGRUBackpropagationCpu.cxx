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
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestGRUBackpropagation.h"
#include "TROOT.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;

int main() {
   std::cout << "Testing GRU backward pass\n";

   //ROOT::EnableImplicitMT(1);
   
   using Scalar_t = Double_t;

   // timesteps, batchsize, statesize, inputsize  { fixed input, with dense layer, with extra GRU }
   testGRUBackpropagation<TCpu<Scalar_t>>(1, 2, 1, 10, 1e-5);

   testGRUBackpropagation<TCpu<Scalar_t>>(1, 2, 3, 2, 1e-5); 

   testGRUBackpropagation<TCpu<Scalar_t>>(2, 3, 4, 5, 1e-5);

   testGRUBackpropagation<TCpu<Scalar_t>>(4, 2, 10, 5, 1e-5);

   testGRUBackpropagation<TCpu<Scalar_t>>(5, 64, 10, 5, 1e-5);
   // using a fixed input 
   testGRUBackpropagation<TCpu<Scalar_t>>(3, 1, 10, 5, 1e-5, {true});
   // with a dense layer 
   testGRUBackpropagation<TCpu<Scalar_t>>(4, 32, 10, 20, 1e-5, {false, true});
   // with an additional GRU layer 
   testGRUBackpropagation<TCpu<Scalar_t>>(4, 32, 10, 5, 1e-5, {false, true, true});


   return 0;
}
