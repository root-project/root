// @(#)root/tmva $Id$
// Author: Saurav Shekhar

/*************************************************************************
 * Copyright (C) 2017, Saurav Shekhar                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Generic tests of the RNNLayer Forward pass                     //
////////////////////////////////////////////////////////////////////

#ifndef TMVA_TEST_DNN_TEST_RNN_TEST_FULL
#define TMVA_TEST_DNN_TEST_RNN_TEST_FULL

#include <iostream>
#include <vector>

#include "../Utility.h"

#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DeepNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;

/* Generate a full recurrent neural net
 * like a word generative model */
//______________________________________________________________________________
template <typename Architecture>
auto testFullRNN(size_t batchSize, size_t stateSize, 
                 size_t inputSize, size_t outputSize)
-> void
{
   using Matrix_t   = typename Architecture::Matrix_t;
   using Tensor_t   = std::vector<Matrix_t>;
   using RNNLayer_t = TBasicRNNLayer<Architecture>; 
   using FCLayer_t  = TDenseLayer<Architecture>;
   using Net_t      = TDeepNet<Architecture>;
   using Scalar_t   = typename Architecture::Scalar_t; 
 
   // check, denselayer takes only first one as input, 
   // so make sure time = 1, in the current case
   size_t timeSteps = 1;
   std::vector<TMatrixT<Double_t>> XRef(timeSteps, TMatrixT<Double_t>(batchSize, inputSize));    // T x B x D
   std::vector<TMatrixT<Double_t>> YRef(timeSteps, TMatrixT<Double_t>(batchSize, outputSize));    // T x B x O
   Tensor_t XArch;
   for (size_t i = 0; i < timeSteps; ++i) {
      randomMatrix(XRef[i]);
      std::cerr << "Copying output into input\n";
      YRef[i] = XRef[i];
      XArch.emplace_back(XRef[i]);
   }

   Net_t rnn(batchSize, timeSteps, batchSize, inputSize, 0, 0, 0, ELossFunction::kMeanSquaredError, EInitialization::kGauss);
   RNNLayer_t* layer = rnn.AddBasicRNNLayer(batchSize, stateSize, inputSize, timeSteps);
   FCLayer_t* classifier = rnn.AddDenseLayer(outputSize, EActivationFunction::kIdentity); 

   Matrix_t W(batchSize, 1);
   for (size_t i = 0; i < batchSize; ++i) W(i, 0) = 1.0;

   size_t iter = 0;
   while (iter++ < 10) {
      rnn.Forward(XArch);
      Scalar_t loss = rnn.Loss(XRef, YRef[0], W, false);

      for (size_t i = 0; i < inputSize/2; ++i) std::cout << XRef[0](0, i) << " "; std::cout << "\n";
      for (size_t i = 0; i < inputSize/2; ++i) std::cout << rnn.GetLayers().back()->GetOutputAt(0)(0, i) << " "; std::cout << "\n";
      std::cout << "The loss is: " << loss << std::endl;

      rnn.Backward(XRef, YRef[0], W);
      rnn.Update(1.0);
   }
}


#endif
