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
// Generic tests of the RNNLayer Backward pass                    //
////////////////////////////////////////////////////////////////////

#ifndef TMVA_TEST_DNN_TEST_RNN_TEST_BWDPASS_H
#define TMVA_TEST_DNN_TEST_RNN_TEST_BWDPASS_H

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
      for (Int_t i = 0; i < A[l].GetNrows(); ++i) {
        for (Int_t j = 0; j < A[l].GetNcols(); ++j) {
            std::cout << A[l](i, j) << " ";
        }
        std::cout << "\n";
      }
      std::cout << "********\n";
  } 
}

/*! Compute the loss of the net as a function of the weight at index (i,j) in
 *  layer l. dx is added as an offset to the current value of the weight. */
//______________________________________________________________________________
template <typename Architecture>
auto evaluate_net_weight(TDeepNet<Architecture> &net, std::vector<typename Architecture::Matrix_t> & X,
                         const typename Architecture::Matrix_t &Y, const typename Architecture::Matrix_t &W, size_t l,
                         size_t k, size_t i, size_t j, typename Architecture::Scalar_t dx) ->
   typename Architecture::Scalar_t
{
    using Scalar_t = typename Architecture::Scalar_t;

    net.GetLayerAt(l)->GetWeightsAt(k).operator()(i,j) += dx;
    Scalar_t res = net.Loss(X, Y, W, false, false);
    net.GetLayerAt(l)->GetWeightsAt(k).operator()(i,j) -= dx;
    return res;
}

/*! Compute the loss of the net as a function of the weight at index i in
 *  layer l. dx is added as an offset to the current value of the weight. */
//______________________________________________________________________________
template <typename Architecture>
auto evaluate_net_bias(TDeepNet<Architecture> &net, std::vector<typename Architecture::Matrix_t> & X,
                       const typename Architecture::Matrix_t &Y, const typename Architecture::Matrix_t &W, size_t l,
                       size_t k, size_t i, typename Architecture::Scalar_t dx) -> typename Architecture::Scalar_t
{
    using Scalar_t = typename Architecture::Scalar_t;

    net.GetLayerAt(l)->GetBiasesAt(k).operator()(i,0) += dx;
    Scalar_t res = net.Loss(X, Y, W, false, false);
    net.GetLayerAt(l)->GetBiasesAt(k).operator()(i,0) -= dx;
    return res;
}

/*! Generate a DeepNet, test backward pass */
//______________________________________________________________________________
template <typename Architecture>
auto testRecurrentBackpropagationWeights(size_t timeSteps, size_t batchSize, size_t stateSize, 
                                         size_t inputSize, typename Architecture::Scalar_t dx)
-> Double_t
{
   using Matrix_t   = typename Architecture::Matrix_t;
   using Tensor_t   = std::vector<Matrix_t>;
   using RNNLayer_t = TBasicRNNLayer<Architecture>; 
   using Net_t      = TDeepNet<Architecture>;
   using Scalar_t = typename Architecture::Scalar_t;

   std::vector<TMatrixT<Double_t>> XRef(batchSize, TMatrixT<Double_t>(timeSteps, inputSize));    // T x B x D
   Tensor_t XArch;
   //for (size_t i = 0; i < batchSize; ++i) XArch.emplace_back(timeSteps, inputSize); // B x T x D
   for (size_t i = 0; i < batchSize; ++i) {
      randomMatrix(XRef[i]);
      XArch.emplace_back(XRef[i]);
   }

   Matrix_t Y(batchSize, timeSteps * stateSize), weights(batchSize, 1);
   randomMatrix(Y);
   fillMatrix(weights, 1.0);

   Net_t rnn(batchSize, batchSize, timeSteps, inputSize, 0, 0, 0, ELossFunction::kMeanSquaredError, EInitialization::kGauss);
   RNNLayer_t* layer = rnn.AddBasicRNNLayer(stateSize, inputSize, timeSteps);
   rnn.AddReshapeLayer(1, timeSteps, stateSize, true);

   rnn.Initialize();
   rnn.Forward(XArch);
   rnn.Backward(XArch, Y, weights);
  
   Scalar_t maximum_error = 0.0;

   // Weights Input, k = 0
   auto &Wi = layer->GetWeightsAt(0);
   auto &dWi = layer->GetWeightGradientsAt(0);
   for (Int_t i = 0; i < Wi.GetNrows(); ++i) {
      for (Int_t j = 0; j < Wi.GetNcols(); ++j) {
         auto f = [&rnn, &XArch, &Y, &weights, i, j](Scalar_t x) {
             return evaluate_net_weight(rnn, XArch, Y, weights, 0, 0, i, j, x);
         };
         Scalar_t dy = finiteDifference(f, dx) / (2.0 * dx);
         Scalar_t dy_ref = dWi(i, j);

         // Compute the relative error if dy != 0.
         Scalar_t error;
         if (std::fabs(dy_ref) > 1e-15) {
            error = std::fabs((dy - dy_ref) / dy_ref);
         } else {
            error = std::fabs(dy - dy_ref);
         }

         maximum_error = std::max(error, maximum_error);
      }
   }

   std::cout << "\rTesting weight input gradients:      ";
   std::cout << "maximum relative error: " << print_error(maximum_error) << std::endl;

   // Weights State, k = 1
   Scalar_t smaximum_error = 0.0;
   auto &Ws = layer->GetWeightsAt(1);
   auto &dWs = layer->GetWeightGradientsAt(1);
   for (Int_t i = 0; i < Ws.GetNrows(); ++i) {
      for (Int_t j = 0; j < Ws.GetNcols(); ++j) {
         auto f = [&rnn, &XArch, &Y, &weights, i, j](Scalar_t x) {
             return evaluate_net_weight(rnn, XArch, Y, weights, 0, 1, i, j, x);
         };
         Scalar_t dy = finiteDifference(f, dx) / (2.0 * dx);
         Scalar_t dy_ref = dWs(i, j);

         // Compute the relative error if dy != 0.
         Scalar_t error;
         if (std::fabs(dy_ref) > 1e-15) {
            error = std::fabs((dy - dy_ref) / dy_ref);
         } else {
            error = std::fabs(dy - dy_ref);
         }

         smaximum_error = std::max(error, smaximum_error);
      }
   }

   std::cout << "\rTesting weight state gradients:      ";
   std::cout << "maximum relative error: " << print_error(smaximum_error) << std::endl;

   return std::max(maximum_error, smaximum_error);
}

/*! Generate a DeepNet, test backward pass */
//______________________________________________________________________________
template <typename Architecture>
auto testRecurrentBackpropagationBiases(size_t timeSteps, size_t batchSize, size_t stateSize, 
                                        size_t inputSize, typename Architecture::Scalar_t dx)
-> Double_t
{
   using Matrix_t   = typename Architecture::Matrix_t;
   using Tensor_t   = std::vector<Matrix_t>;
   using RNNLayer_t = TBasicRNNLayer<Architecture>; 
   using Net_t      = TDeepNet<Architecture>;
   using Scalar_t = typename Architecture::Scalar_t;

   std::vector<TMatrixT<Double_t>> XRef(batchSize, TMatrixT<Double_t>(timeSteps, inputSize));    // T x B x D
   Tensor_t XArch;
   //for (size_t i = 0; i < batchSize; ++i) XArch.emplace_back(timeSteps, inputSize); // B x T x D
   for (size_t i = 0; i < batchSize; ++i) {
      randomMatrix(XRef[i]);
      XArch.emplace_back(XRef[i]);
   }

   Matrix_t Y(batchSize, stateSize), weights(batchSize, 1);
   randomMatrix(Y);
   fillMatrix(weights, 1.0);

   Net_t rnn(batchSize, batchSize, timeSteps, inputSize, 0, 0, 0, ELossFunction::kMeanSquaredError, EInitialization::kGauss);
   RNNLayer_t* layer = rnn.AddBasicRNNLayer(stateSize, inputSize, timeSteps);
   rnn.AddReshapeLayer(1, timeSteps, stateSize, true);

   rnn.Initialize();
   rnn.Forward(XArch);
   rnn.Backward(XArch, Y, weights);
  
   Scalar_t maximum_error = 0.0;

   auto &B = layer->GetBiasesAt(0);
   auto &dB = layer->GetBiasGradientsAt(0);
   for (Int_t i = 0; i < B.GetNrows(); ++i) {
      auto f = [&rnn, &XArch, &Y, &weights, i](Scalar_t x) {
          return evaluate_net_bias(rnn, XArch, Y, weights, 0, 0, i, x);
      };
      Scalar_t dy = finiteDifference(f, dx) / (2.0 * dx);
      Scalar_t dy_ref = dB(i, 0);

      // Compute the relative error if dy != 0.
      Scalar_t error;
      if (std::fabs(dy_ref) > 1e-15) {
         error = std::fabs((dy - dy_ref) / dy_ref);
      } else {
         error = std::fabs(dy - dy_ref);
      }

      maximum_error = std::max(error, maximum_error);
   }

   std::cout << "\rTesting bias gradients:      ";
   std::cout << "maximum relative error: " << print_error(maximum_error) << std::endl;

   return maximum_error;
}

#endif
