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
#include "Math/Functor.h"
#include "Math/RichardsonDerivator.h"

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

template <typename Architecture>
auto printTensor(const typename Architecture::Matrix_t &A, const std::string name = "matrix")
-> void
{
  std::cout << name << "\n";
  for (Int_t i = 0; i < A.GetNrows(); ++i) {
    for (Int_t j = 0; j < A.GetNcols(); ++j) {
        std::cout << A(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "********\n";
}

/*! Compute the loss of the net as a function of the weight at index (i,j) in
 *  layer l. dx is added as an offset to the current value of the weight. */
//______________________________________________________________________________
template <typename Architecture>
auto evaluate_net_weight(TDeepNet<Architecture> &net, std::vector<typename Architecture::Matrix_t> & X,
                         const typename Architecture::Matrix_t &Y, const typename Architecture::Matrix_t &W, size_t l,
                         size_t k, size_t i, size_t j, typename Architecture::Scalar_t xvalue) ->
   typename Architecture::Scalar_t
{
    using Scalar_t = typename Architecture::Scalar_t;

    Scalar_t prev_value = net.GetLayerAt(l)->GetWeightsAt(k).operator()(i,j);
    net.GetLayerAt(l)->GetWeightsAt(k).operator()(i,j) = xvalue;
    Scalar_t res = net.Loss(X, Y, W, false, false);
    net.GetLayerAt(l)->GetWeightsAt(k).operator()(i,j) = prev_value;
    return res;
}

/*! Compute the loss of the net as a function of the weight at index i in
 *  layer l. dx is added as an offset to the current value of the weight. */
//______________________________________________________________________________
template <typename Architecture>
auto evaluate_net_bias(TDeepNet<Architecture> &net, std::vector<typename Architecture::Matrix_t> & X,
                       const typename Architecture::Matrix_t &Y, const typename Architecture::Matrix_t &W, size_t l,
                       size_t k, size_t i, typename Architecture::Scalar_t xvalue) -> typename Architecture::Scalar_t
{
    using Scalar_t = typename Architecture::Scalar_t;
 
    Scalar_t prev_value = net.GetLayerAt(l)->GetBiasesAt(k).operator()(i,0);
    net.GetLayerAt(l)->GetBiasesAt(k).operator()(i,0) = xvalue;
    Scalar_t res = net.Loss(X, Y, W, false, false);
    net.GetLayerAt(l)->GetBiasesAt(k).operator()(i,0) = prev_value; 
    return res;
}

/*! Generate a DeepNet, test backward pass */
//______________________________________________________________________________
template <typename Architecture>
auto testRecurrentBackpropagation(size_t timeSteps, size_t batchSize, size_t stateSize, 
                                  size_t inputSize, typename Architecture::Scalar_t dx = 1.E-5, bool randomInput = true)
-> Double_t
{
   bool debug = false; 
   
   using Matrix_t   = typename Architecture::Matrix_t;
   using Tensor_t   = std::vector<Matrix_t>;
   using RNNLayer_t = TBasicRNNLayer<Architecture>; 
   using Net_t      = TDeepNet<Architecture>;
   using Scalar_t = typename Architecture::Scalar_t;

   //std::vector<Matrix_t<Double_t>> XRef(batchSize, Matrix_t<Double_t>(timeSteps, inputSize));    // B x T x D
   Tensor_t XArch;  // B x T x D
   for (size_t i = 0; i < batchSize; ++i) {
      XArch.emplace_back(timeSteps, inputSize);
   }

   // for random input (default) 
   if (randomInput) { 
   for (size_t i = 0; i < batchSize; ++i) {
         for (size_t l = 0; l < XArch[i].GetNrows(); ++l) {
            for (size_t m = 0; m < XArch[i].GetNcols(); ++m) {
               //XArch[i](l, m) = i + l + m;
               XArch[i](l, m) = gRandom->Uniform(-1,1);
            }
         } 
      }
   }
   else { 
      debug = true; 
      R__ASSERT(inputSize <= 6); 
      R__ASSERT(timeSteps <= 3); 
      R__ASSERT(batchSize <= 1); 
      double x_input[] = {-1,   1,  -2,  2, -3,  3 ,
                          -0.5, 0.5,-0.8,0.9, -2, 1.5,
                          -0.2, 0.1,-0.5,0.4, -1, 1.};

      TMatrixD Input(3,6,x_input);
      for (size_t i = 0; i < batchSize; ++i) {
         auto & mat = XArch[i];
         // time 0
         for (int l = 0; l < timeSteps; ++l) {
            for (int m = 0; m < inputSize; ++m) {
               mat(l,m) = Input(l,m);
            }
         }
      }
      gRandom->SetSeed(1); // for weights initizialization
   }
   printTensor<Architecture>(XArch,"input"); 
   
   Matrix_t Y(batchSize, timeSteps * stateSize), weights(batchSize, 1);
   //randomMatrix(Y);
   for (size_t i = 0; i < Y.GetNrows(); ++i) {
     for (size_t j = 0; j < Y.GetNcols(); ++j) {
        Y(i, j) = 1;// (i + j)/2.0 - 0.75;
     }
   }
   fillMatrix(weights, 1.0);

   std::cout << "Testing Weight Backprop using RNN with batchsize = " << batchSize << " input = " << inputSize << " state = " << stateSize << " time = " << timeSteps << std::endl;
   Net_t rnn(batchSize, batchSize, timeSteps, inputSize, 0, 0, 0, ELossFunction::kMeanSquaredError, EInitialization::kGauss);
   RNNLayer_t* layer = rnn.AddBasicRNNLayer(stateSize, inputSize, timeSteps);
   //layer->Print(); 
   rnn.AddReshapeLayer(1, timeSteps, stateSize, true);

   rnn.Initialize();


   auto & wi = layer->GetWeights()[0];
   for (int i = 0; i < stateSize; ++i) { 
      for (int j = 0; j < inputSize; ++j) { 
         wi(i,j) =  gRandom->Uniform(-1,1);
      }
         wi(i,i) = 1.; 
   }
   
   auto & wh = layer->GetWeights()[1];
   for (int i = 0; i < stateSize; ++i) { 
      for (int j = 0; j < stateSize; ++j) { 
         wh(i,j) = gRandom->Uniform(-1,1);
      }
         wh(i,i) = 0.5; 
   }
   auto & b = layer->GetBiases()[0];
   for (int i = 0; i < b.GetNrows(); ++i) { 
      for (int j = 0; j < b.GetNcols(); ++j) { 
         b(i,j) = gRandom->Uniform(-0.5,0.5);
      }
   }

   
   rnn.Forward(XArch);
   rnn.Backward(XArch, Y, weights);

   if (debug)  {
      auto & out = layer->GetOutput();
      printTensor<Architecture>(out,"output"); 
   }

   
   Scalar_t maximum_error = 0.0;
   std::string maxerrorType; 

   ROOT::Math::RichardsonDerivator deriv;
   

   // Weights Input, k = 0
   auto &Wi = layer->GetWeightsAt(0);
   auto &dWi = layer->GetWeightGradientsAt(0);
   for (Int_t i = 0; i < Wi.GetNrows(); ++i) {
      for (Int_t j = 0; j < Wi.GetNcols(); ++j) {
         auto f = [&rnn, &XArch, &Y, &weights, i, j](Scalar_t x) {
             return evaluate_net_weight(rnn, XArch, Y, weights, 0, 0, i, j, x);
         };
         ROOT::Math::Functor1D func(f);
         double dy = deriv.Derivative1(func, Wi(i,j), 1.E-5);
         Scalar_t dy_ref = dWi(i, j);

         // Compute the relative error if dy != 0.
         Scalar_t error;
         std::string errorType; 
         if (std::fabs(dy_ref) > 1e-15) {
            error = std::fabs((dy - dy_ref) / dy_ref);
            errorType = "relative";
         } else {
            error = std::fabs(dy - dy_ref);
            errorType = "absolute";
         }

         if (debug) std::cout << "Weight-input gradient (" << i << "," << j << ") : (comp, ref) " << dy << " , " << dy_ref << std::endl;

         if (error >= maximum_error) {
            maximum_error = error;
            maxerrorType = errorType; 
         }
      }
   }

   std::cout << "\rTesting weight input gradients:      ";
   std::cout << "maximum error (" << maxerrorType << "): "  << print_error(maximum_error) << std::endl;

   /// testing weight state gradient
   
   // Weights State, k = 1
   maximum_error = 0; 
   auto &Ws = layer->GetWeightsAt(1);
   auto &dWs = layer->GetWeightGradientsAt(1);
   for (Int_t i = 0; i < Ws.GetNrows(); ++i) {
      for (Int_t j = 0; j < Ws.GetNcols(); ++j) {
         auto f = [&rnn, &XArch, &Y, &weights, i, j](Scalar_t x) {
             return evaluate_net_weight(rnn, XArch, Y, weights, 0, 1, i, j, x);
         };
         ROOT::Math::Functor1D func(f);
         double dy = deriv.Derivative1(func, Ws(i,j), 1.E-5);
         Scalar_t dy_ref = dWs(i, j);

         // Compute the relative error if dy != 0.
         Scalar_t error;
         std::string errorType;  
         if (std::fabs(dy_ref) > 1e-15) {
            error = std::fabs((dy - dy_ref) / dy_ref);
            errorType = "relative";
         } else {
            error = std::fabs(dy - dy_ref);
            errorType = "absolute";
         }

         if ( error >= maximum_error) {
            maximum_error = error; 
            maxerrorType = errorType; 
         }
         if (debug) std::cout << "Weight-state gradient (" << i << "," << j << ") : (comp, ref) " << dy << " , " << dy_ref << std::endl;
      }
   }

   std::cout << "\rTesting weight state gradients:      ";
   std::cout << "maximum error (" << maxerrorType << "): "  << print_error(maximum_error) << std::endl;

   // testing bias gradients
   maximum_error = 0; 
   auto &B = layer->GetBiasesAt(0);
   auto &dB = layer->GetBiasGradientsAt(0);
   for (Int_t i = 0; i < B.GetNrows(); ++i) {
      auto f = [&rnn, &XArch, &Y, &weights, i](Scalar_t x) {
          return evaluate_net_bias(rnn, XArch, Y, weights, 0, 0, i, x);
      };
      ROOT::Math::Functor1D func(f);
      double dy = deriv.Derivative1(func, B(i,0), 1.E-5);
      Scalar_t dy_ref = dB(i, 0);

      // Compute the relative error if dy != 0.
      Scalar_t error;
      std::string errorType;  
      if (std::fabs(dy_ref) > 1e-15) {
         error = std::fabs((dy - dy_ref) / dy_ref);
         errorType = "relative";
      } else {
         error = std::fabs(dy - dy_ref);
         errorType = "absolute";
      }

      if ( error >= maximum_error) {
            maximum_error = error; 
            maxerrorType = errorType; 
      }
      if (debug) std::cout << "Bias gradient (" << i << ") : (comp, ref) " << dy << " , " << dy_ref << std::endl;
   }

   std::cout << "\rTesting bias gradients:      ";
   std::cout << "maximum error (" << maxerrorType << "): "  << print_error(maximum_error) << std::endl;


   //return std::max(maximum_error, smaximum_error);
   return maximum_error; 
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

   std::cout << "Testing Bias Backprop using RNN with batchsize = " << batchSize << " input = " << inputSize << " state = " << stateSize << " time = " << timeSteps << std::endl;

   Net_t rnn(batchSize, batchSize, timeSteps, inputSize, 0, 0, 0, ELossFunction::kMeanSquaredError, EInitialization::kGauss);
   RNNLayer_t* layer = rnn.AddBasicRNNLayer(stateSize, inputSize, timeSteps);
   rnn.AddReshapeLayer(1, timeSteps, stateSize, true);

   rnn.Initialize();
   rnn.Forward(XArch);
   rnn.Backward(XArch, Y, weights);
  
   Scalar_t maximum_error = 0.0;
   std::string merrorType;  
      
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
      std::string errorType;  
      if (std::fabs(dy_ref) > 1e-15) {
         error = std::fabs((dy - dy_ref) / dy_ref);
         errorType = "relative";
      } else {
         error = std::fabs(dy - dy_ref);
         errorType = "absolute";
      }

      if ( error >= maximum_error) {
            maximum_error = error; 
            merrorType = errorType; 
      }
      std::cout << "Bias gradient (" << i << ") : (comp, ref) " << dy << " , " << dy_ref << std::endl;
      //maximum_error = std::max(error, maximum_error);
   }

   std::cout << "\rTesting bias gradients:      ";
   std::cout << "maximum error (" << merrorType << "): "  << print_error(maximum_error) << std::endl;

   return maximum_error;
}

#endif
