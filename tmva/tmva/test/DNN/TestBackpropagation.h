// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Generic tests of the backpropagation algorithm.                //
//                                                                //
// All tests randomly generate a net with identity activation     //
// functions, i.e.  which is completely linear and then tests the //
// computed gradients for each layer using numerical              //
// derivation. The restriction to linear nets is to avoid the     //
// required division by the finite difference interval used to    //
// approximate the numerical derivatives, which would otherwise   //
// cause precision loss.                                          //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/Net.h"
#include "Utility.h"

using namespace TMVA::DNN;

/*! Compute the loss of the net as a function of the weight at index (i,j) in
 *  layer l. dx is added as an offset to the current value of the weight. */
//______________________________________________________________________________
template <typename Architecture>
auto evaluate_net_weight(TNet<Architecture> &net, typename Architecture::Matrix_t &X,
                         const typename Architecture::Matrix_t &Y, const typename Architecture::Matrix_t &W, size_t l,
                         size_t i, size_t j, typename Architecture::Scalar_t dx) -> typename Architecture::Scalar_t
{
    using Scalar_t = typename Architecture::Scalar_t;

    net.GetLayer(l).GetWeights().operator()(i,j) += dx;
    Scalar_t res = net.Loss(X, Y, W);
    net.GetLayer(l).GetWeights().operator()(i,j) -= dx;
    return res;
}

/*! Compute the loss of the net as a function of the weight at index i in
 *  layer l. dx is added as an offset to the current value of the weight. */
//______________________________________________________________________________
template <typename Architecture>
auto evaluate_net_bias(TNet<Architecture> &net, typename Architecture::Matrix_t &X,
                       const typename Architecture::Matrix_t &Y, const typename Architecture::Matrix_t &W, size_t l,
                       size_t i, typename Architecture::Scalar_t dx) -> typename Architecture::Scalar_t
{
    using Scalar_t = typename Architecture::Scalar_t;

    net.GetLayer(l).GetBiases().operator()(i,0) += dx;
    Scalar_t res = net.Loss(X, Y, W);
    net.GetLayer(l).GetBiases().operator()(i,0) -= dx;
    return res;
}

/*! Generate a random net, perform forward and backward propagation and check
 *  the weight gradients using numerical differentiation. Returns the maximum
 *  relative gradient error and also prints it to stdout. */
//______________________________________________________________________________
template <typename Architecture>
auto testBackpropagationWeightsLinear(typename Architecture::Scalar_t dx)
-> typename Architecture::Scalar_t
{
   using Scalar_t = typename Architecture::Scalar_t;
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t = TNet<Architecture>;

   Net_t net(50, 50, ELossFunction::kMeanSquaredError);

   // Random net.
   constructRandomLinearNet(net);
   net.Initialize(EInitialization::kGauss);

   // Random training data.
   Matrix_t X(50, 50), Y(50, net.GetOutputWidth()), weights(50, 1);
   randomMatrix(X);
   randomMatrix(Y);
   fillMatrix(weights, 1.0);

   net.Forward(X);
   net.Backward(X, Y, weights);

   Scalar_t maximum_error = 0.0;

   // Compute derivatives for all weights using finite differences and
   // compare to result obtained from backpropagation.
   for (size_t l = 0; l < net.GetDepth(); l++) {
      std::cout << "\rTesting weight gradients:      layer: " << l << " / " << net.GetDepth();
      std::cout << std::flush;
      auto &layer = net.GetLayer(l);
      auto &W = layer.GetWeightGradients();

      for (size_t i = 0; i < layer.GetWidth(); i++) {
         for (size_t j = 0; j < layer.GetInputWidth(); j++) {
            auto f = [&net, &X, &Y, &weights, l, i, j](Scalar_t x) {
               return evaluate_net_weight(net, X, Y, weights, l, i, j, x);
            };
            Scalar_t dy = finiteDifference(f, dx) / (2.0 * dx);
            Scalar_t dy_ref = W(i, j);

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
   }

   std::cout << "\rTesting weight gradients:      ";
   std::cout << "maximum relative error: " << print_error(maximum_error) << std::endl;
   return maximum_error;
}

/*! Generate a random, linear net, perform forward and backward propagation with
 *  L1 regularization and check the weight gradients using numerical
 *  differentiation. Returns the maximum relative gradient error and
 *  also prints it to stdout. */
//______________________________________________________________________________
template <typename Architecture>
auto testBackpropagationL1Regularization(typename Architecture::Scalar_t dx)
-> typename Architecture::Scalar_t
{
   using Scalar_t = typename Architecture::Scalar_t;
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t    = TNet<Architecture>;

   Net_t net(50, 50, ELossFunction::kMeanSquaredError, ERegularization::kL1, 0.1);

   // Random net.
   constructRandomLinearNet(net);
   net.Initialize(EInitialization::kGauss);

   // Random training data.
   Matrix_t X(50, 50), Y(50, net.GetOutputWidth()), weights(50, 1);
   randomMatrix(X);
   randomMatrix(Y);
   fillMatrix(weights, 1.0);

   net.Forward(X);
   net.Backward(X, Y, weights);

   Scalar_t maximum_error = 0.0;

   // Compute derivatives for all weights using finite differences and
   // compare to result obtained from backpropagation.
   for (size_t l = 0; l < net.GetDepth(); l++)
   {
      std::cout << "\rTesting weight gradients (L1): layer: "
                << l << " / " << net.GetDepth();
      std::cout << std::flush;
      auto & layer = net.GetLayer(l);
      auto & W     = layer.GetWeights();
      auto & dW    = layer.GetWeightGradients();

      for (size_t i = 0; i < layer.GetWidth(); i++) {
         for (size_t j = 0; j < layer.GetInputWidth(); j++) {
            // Avoid running into the non-derivable point at 0.0.
            if (std::abs(W(i,j)) > dx) {
               auto f = [&net, &X, &Y, &weights, l, i, j](Scalar_t x) {
                  return evaluate_net_weight(net, X, Y, weights, l, i, j, x);
               };
               Scalar_t dy     = finiteDifference(f, dx) / (2.0 * dx);
               Scalar_t dy_ref = dW(i,j);

               // Compute the relative error if dy != 0.
               Scalar_t error;
               if (std::fabs(dy_ref) > 1e-15)
               {
                  error = std::fabs((dy - dy_ref) / dy_ref);
               }
               else
               {
                  error = std::fabs(dy - dy_ref);
               }

               maximum_error = std::max(error, maximum_error);
            }
         }
      }
   }

   std::cout << "\rTesting weight gradients (L1): ";
   std::cout << "maximum relative error: " << print_error(maximum_error) << std::endl;
   return maximum_error;
}

/*! Generate a random, linear net, perform forward and backward propagation with
 *  L2 regularization and check the weight gradients using numerical
 *  differentiation. Returns the maximum relative gradient error and
 *  also prints it to stdout. */
//______________________________________________________________________________
template <typename Architecture>
auto testBackpropagationL2Regularization(typename Architecture::Scalar_t dx)
-> typename Architecture::Scalar_t
{
   using Scalar_t = typename Architecture::Scalar_t;
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t    = TNet<Architecture>;

   Net_t net(50, 50, ELossFunction::kMeanSquaredError, ERegularization::kL2, 0.1);

   // Random net.
   constructRandomLinearNet(net);
   net.Initialize(EInitialization::kGauss);

   // Random training data.
   Matrix_t X(50, 50), Y(50, net.GetOutputWidth()), weights(50, 1);
   randomMatrix(X);
   randomMatrix(Y);
   fillMatrix(weights, 1.0);

   net.Forward(X);
   net.Backward(X, Y, weights);

   Scalar_t maximum_error = 0.0;

   // Compute derivatives for all weights using finite differences and
   // compare to result obtained from backpropagation.
   for (size_t l = 0; l < net.GetDepth(); l++)
   {
      std::cout << "\rTesting weight gradients (L2): layer: "
                << l << " / " << net.GetDepth();
      std::cout << std::flush;
      auto & layer = net.GetLayer(l);
      auto & W     = layer.GetWeightGradients();

      for (size_t i = 0; i < layer.GetWidth(); i++)
      {
         for (size_t j = 0; j < layer.GetInputWidth(); j++)
         {
            auto f = [&net, &X, &Y, &weights, l, i, j](Scalar_t x) {
               return evaluate_net_weight(net, X, Y, weights, l, i, j, x);
            };
            Scalar_t dy     = finiteDifference(f, dx) / (2.0 * dx);
            Scalar_t dy_ref = W(i,j);

            // Compute the relative error if dy != 0.
            Scalar_t error;
            if (std::fabs(dy_ref) > 1e-15)
            {
               error = std::fabs((dy - dy_ref) / dy_ref);
            }
            else
            {
               error = std::fabs(dy - dy_ref);
            }

            maximum_error = std::max(error, maximum_error);
         }
      }
   }

   std::cout << "\rTesting weight gradients (L2): ";
   std::cout << "maximum relative error: " << print_error(maximum_error) << std::endl;
   return maximum_error;
}

/*! Generate a random net, perform forward and backward propagation and check
 *  the bias gradients using numerical differentiation. Returns the maximum
 *  relative gradient error and also prints it to stdout. */
//______________________________________________________________________________
template <typename Architecture>
auto testBackpropagationBiasesLinear(typename Architecture::Scalar_t dx)
-> typename Architecture::Scalar_t
{
   using Net_t    = TNet<Architecture>;
   using Scalar_t   = typename Architecture::Scalar_t;
   using Matrix_t = typename Architecture::Matrix_t;


   Net_t net(50, 50, ELossFunction::kMeanSquaredError);

   // Random net.
   constructRandomLinearNet(net);
   net.Initialize(EInitialization::kGauss);

   // Random training data.
   Matrix_t X(50, 50), Y(50, net.GetOutputWidth()), weights(50, 1);
   randomMatrix(X);
   randomMatrix(Y);
   fillMatrix(weights, 1.0);

   net.Forward(X);
   net.Backward(X, Y, weights);

   Scalar_t maximum_error = 0.0;

   // Compute derivatives for all bias terms using finite differences and
   // compare to result obtained from backpropagation.
   for (size_t l = 0; l < net.GetDepth(); l++)
   {
      std::cout << "\rTesting bias gradients:       layer: "
                << l << " / " << net.GetDepth();
      std::cout << std::flush;
      auto & layer = net.GetLayer(l);
      auto & dtheta = layer.GetBiasGradients();

      for (size_t i = 0; i < layer.GetWidth(); i++)
      {
         auto f = [&net, &X, &Y, &weights, l, i](Scalar_t x) { return evaluate_net_bias(net, X, Y, weights, l, i, x); };
         Scalar_t dy     = finiteDifference(f, dx);
         Scalar_t dy_ref = dtheta(i,0) * 2.0 * dx;

         // Compute the relative error if dy != 0.
         Scalar_t error;
         if (std::fabs(dy_ref) > 1e-10)
         {
            error = std::fabs((dy - dy_ref) / dy_ref);
         }
         else
         {
            error = std::fabs(dy - dy_ref);
         }

         maximum_error = std::max(error, maximum_error);
      }
   }

   std::cout << "\rTesting bias gradients:        ";
   std::cout << "maximum relative error: " << print_error(maximum_error) << std::endl;
   return maximum_error;
}
