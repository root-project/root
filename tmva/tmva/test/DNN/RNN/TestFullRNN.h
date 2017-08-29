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
#include "TMVA/DNN/Net.h"

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

template <typename Architecture>
auto printMatrix(const typename Architecture::Matrix_t &A, const std::string name = "matrix")
-> void
{
  std::cout << name << "\n";
  for (size_t i = 0; i < A.GetNrows(); ++i) {
    for (size_t j = 0; j < A.GetNcols(); ++j) {
        std::cout << A(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "********\n";
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
   using Reshape_t  = TReshapeLayer<Architecture>;
   using Net_t      = TDeepNet<Architecture>;
   using Scalar_t   = typename Architecture::Scalar_t; 
   using MLP_t      = TNet<Architecture>;    
 
   // check, denselayer takes only first one as input, 
   // so make sure time = 1, in the current case
   size_t timeSteps = 1;
   std::vector<TMatrixT<Double_t>> XRef(batchSize, TMatrixT<Double_t>(timeSteps, inputSize));    // B x T x D
   //TMatrixT<Double_t> YRef(batchSize, outputSize);    // B x O  (D = O)
   Tensor_t XArch;
   Matrix_t YArch(batchSize, outputSize);             // B x O  (D = O)
   for (size_t i = 0; i < batchSize; ++i) {
      randomMatrix(XRef[i]);
      std::cerr << "Copying output into input\n";
      XArch.emplace_back(XRef[i]);
      for (size_t j = 0; j < outputSize; ++j) {
         YArch(i, j) = XArch[i](0, j);
      }
   }

   Net_t rnn(batchSize, batchSize, timeSteps, inputSize, 0, 0, 0, ELossFunction::kMeanSquaredError, EInitialization::kGauss);
   RNNLayer_t* layer = rnn.AddBasicRNNLayer(stateSize, inputSize, timeSteps, false);
   Reshape_t* reshape = rnn.AddReshapeLayer(1, 1, stateSize, true);
   FCLayer_t* classifier = rnn.AddDenseLayer(outputSize, EActivationFunction::kIdentity); 

   Matrix_t W(batchSize, 1);
   for (size_t i = 0; i < batchSize; ++i) W(i, 0) = 1.0;
   rnn.Initialize();

   size_t iter = 0;
   while (iter++ < 50) {
      rnn.Forward(XArch);
      Scalar_t loss = rnn.Loss(YArch, W, false);

      //if (iter % 20 == 0) {
         //for (size_t i = 0; i < inputSize; ++i) std::cout << XRef[0](0, i) << " "; std::cout << "\n";
         //for (size_t i = 0; i < inputSize; ++i) std::cout << rnn.GetLayers().back()->GetOutputAt(0)(0, i) << " "; std::cout << "\n";
      //}
      std::cout << "loss: " << loss << std::endl;

      rnn.Backward(XArch, YArch, W);

      rnn.Update(0.1);

   }
}


#endif
