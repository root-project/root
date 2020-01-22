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

#ifndef TMVA_TEST_DNN_TEST_RNN_TEST_FWDPASS_H
#define TMVA_TEST_DNN_TEST_RNN_TEST_FWDPASS_H

#include <iostream>
#include <vector>

#include "../Utility.h"

#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DeepNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;

// template <typename Architecture>
// auto printTensor1(const std::vector<typename Architecture::Matrix_t> &A, const std::string name = "matrix")
// -> void
// {
//   std::cout << name << "\n";
//   for (size_t l = 0; l < A.size(); ++l) {
//       for (size_t i = 0; i < A[l].GetNrows(); ++i) {
//         for (size_t j = 0; j < A[l].GetNcols(); ++j) {
//             std::cout << A[l](i, j) << " ";
//         }
//         std::cout << "\n";
//       }
//       std::cout << "********\n";
//   }
// }

// template <typename Architecture>
// auto printMatrix1(const typename Architecture::Matrix_t &A, const std::string name = "matrix")
// -> void
// {
//   std::cout << name << "\n";
//   for (size_t i = 0; i < A.GetNrows(); ++i) {
//     for (size_t j = 0; j < A.GetNcols(); ++j) {
//         std::cout << A(i, j) << " ";
//     }
//     std::cout << "\n";
//   }
//   std::cout << "********\n";
// }


/*! Generate a DeepNet, test forward pass */
//______________________________________________________________________________
template <typename Architecture>
auto testForwardPass(size_t timeSteps, size_t batchSize, size_t stateSize,
                               size_t inputSize)
-> Double_t
{
   //using Matrix_t   = typename Architecture::Matrix_t;
   using Tensor_t   = typename Architecture::Tensor_t;
   using RNNLayer_t = TBasicRNNLayer<Architecture>;
   using Net_t      = TDeepNet<Architecture>;

   // std::vector<TMatrixT<Double_t>> XRef(timeSteps, TMatrixT<Double_t>(batchSize, inputSize));    // T x B x D
   Tensor_t XArch (timeSteps, batchSize, inputSize);    // T x B x D
   Tensor_t XRef = XArch;
   Tensor_t arr_XArch(batchSize, timeSteps, inputSize); // B x T x D

   //randomBatch(XArch);
   double x_input[] = {-1, 1, -2, 2, -3, 3, -0.5, 0.5, -0.8, 0.9, -2, 1.5, -0.2, 0.1, -0.5, 0.4, -1, 1.};
   TMatrixD Input(3, 6, x_input);
   for (size_t i = 0; i < batchSize; ++i) {
      for (size_t l = 0; l < timeSteps; ++l) {
         for (size_t m = 0; m < inputSize; ++m) {
            XArch(i, l, m) = Input(l, m);
         }
      }
   }

   Architecture::Rearrange(arr_XArch, XArch);     // B x T x D

   Net_t rnn(batchSize, batchSize, timeSteps, inputSize, 0, 0, 0, ELossFunction::kMeanSquaredError,
             EInitialization::kGlorotUniform);
   RNNLayer_t* layer = rnn.AddBasicRNNLayer(stateSize, inputSize, timeSteps, false, true);

   layer->Initialize();

   auto weightsInput = layer->GetWeightsInput();  // H x D
   TMatrixD weightsState = layer->GetWeightsState();  // H x H
   TMatrixD biases = layer->GetBiasesAt(0);           // H x 1
   TMatrixD state = layer->GetState();                // B x H
   TMatrixD tmp(batchSize, stateSize);

   rnn.Forward(arr_XArch);
   Tensor_t outputArch = layer->GetOutput();    // B x T x H
   Tensor_t arr_outputArch (timeSteps, batchSize, stateSize); // T x B x H

   Architecture::Rearrange(arr_outputArch, outputArch);

   Double_t maximumError = 0.0;
   for (size_t t = 0; t < timeSteps; ++t) {
      tmp.MultT(state, weightsState);
      state.MultT(XRef.At(t).GetMatrix(), weightsInput);
      state += tmp;
      // adding bias
      for (size_t i = 0; i < (size_t) state.GetNrows(); i++) {
         for (size_t j = 0; j < (size_t) state.GetNcols(); j++) {
            state(i,j) += biases(j,0);
         }
      }
      // activation fn
      applyMatrix(state, [](double x){return tanh(x);});
      TMatrixD output = arr_outputArch.At(t).GetMatrix();

      Double_t error = maximumRelativeError(output, state);
      std::cout << "Time " << t << " Error: " << error << "\n";
      maximumError = std::max(error, maximumError);
   }
   return maximumError;
}

#endif
