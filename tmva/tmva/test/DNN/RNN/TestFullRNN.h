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

template <typename Architecture>
auto printTensor(const std::vector<typename Architecture::Matrix_t> &A, const std::string name = "matrix")
-> void
{
  std::cout << name << "\n";
  for (size_t l = 0; l < A.size(); ++l) {
      for (size_t i = 0; i < A[l].GetNrows(); ++i) {
        for (size_t j = 0; j < A[l].GetNcols(); ++j) {
            std::cout << A[l](i, j) << " ";
        }
        std::cout << "\n";
      }
      std::cout << "********\n";
  } 
}

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
   //RNNLayer_t* layer = rnn.AddBasicRNNLayer(stateSize, inputSize, timeSteps, false);
   FCLayer_t* classifier = rnn.AddDenseLayer(outputSize, EActivationFunction::kIdentity); 

   Matrix_t W(batchSize, 1);
   for (size_t i = 0; i < batchSize; ++i) W(i, 0) = 1.0;
   rnn.Initialize();

   //printTensor<Architecture>(classifier->GetWeights(), "dense weights");
   //printTensor<Architecture>(classifier->GetBiases(), "dense biases");
   //printTensor<Architecture>(layer->GetWeights(), "rnn weights");
   //printTensor<Architecture>(layer->GetBiases(), "rnn biases");

   size_t iter = 0;
   while (iter++ < 3) {
      rnn.Forward(XArch);
      //printTensor<Architecture>(layer->GetOutput(), "RNN Output");
      Scalar_t loss = rnn.Loss(YRef[0], W, false);

      for (size_t i = 0; i < inputSize; ++i) std::cout << XRef[0](0, i) << " "; std::cout << "\n";
      for (size_t i = 0; i < inputSize; ++i) std::cout << rnn.GetLayers().back()->GetOutputAt(0)(0, i) << " "; std::cout << "\n";
      std::cout << "The loss is: " << loss << std::endl;

      rnn.Backward(XRef, YRef[0], W);

      printTensor<Architecture>(classifier->GetWeights(), "dense weight");
      printTensor<Architecture>(classifier->GetWeightGradients(), "dense weight grad");
      printTensor<Architecture>(classifier->GetBiases(), "dense bias");
      printTensor<Architecture>(classifier->GetBiasGradients(), "dense bias grad");
      //printTensor<Architecture>(layer->GetWeightGradients(), "rnn weight grad");
      //printTensor<Architecture>(layer->GetBiasGradients(), "rnn bias grad");

      rnn.Update(1.0);
   //printTensor<Architecture>(classifier->GetWeights(), "dense weights");
   //printTensor<Architecture>(classifier->GetBiases(), "dense biases");
   //printTensor<Architecture>(layer->GetWeights(), "rnn weights");
   //printTensor<Architecture>(layer->GetBiases(), "rnn biases");
   }
}


#endif
