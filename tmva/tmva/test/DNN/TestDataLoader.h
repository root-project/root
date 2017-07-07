// @(#)root/tmva/tmva/dnn:$Id$
// Author: Simon Pfreundschuh 12/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////
// Generic test for DataLoader implementations. //
//////////////////////////////////////////////////

#include "TMVA/DNN/Net.h"
#include "TMVA/DNN/DataLoader.h"
#include "Utility.h"

namespace TMVA
{
namespace DNN
{

/** Test that the data loader loads all data in the data set by summing
 *  up all elements batch wise and comparing to the result obtained by summing
 *  over the complete dataset.
 */
//______________________________________________________________________________
template <typename Architecture_t>
auto testSum()
    -> typename Architecture_t::Scalar_t
{
   using Scalar_t     = typename Architecture_t::Scalar_t;
   using Matrix_t     = typename Architecture_t::Matrix_t;
   using DataLoader_t = TDataLoader<MatrixInput_t, Architecture_t>;

   size_t nSamples = 10000;
   TMatrixT<Double_t> X(nSamples,1);
   randomMatrix(X);
   for (size_t i = 0; i < 10000; i++) {
      X(i,0) = i;
   }
   MatrixInput_t input(X, X, X);
   DataLoader_t  loader(input, nSamples, 5, 1, 1);

   Matrix_t XArch(X), Sum(1,1), SumTotal(1,1);
   Scalar_t sum = 0.0, sumTotal = 0.0;

   for (auto b : loader) {
      Architecture_t::SumColumns(Sum, b.GetInput());
      sum += Sum(0, 0);
      Architecture_t::SumColumns(Sum, b.GetOutput());
      sum += Sum(0, 0);
      Architecture_t::SumColumns(Sum, b.GetWeights());
      sum += Sum(0, 0);
   }

   Architecture_t::SumColumns(SumTotal, XArch);
   sumTotal = 3.0 * SumTotal(0, 0);

   return fabs(sumTotal - sum) / sumTotal;
}

/** Test the data loader by loading identical input and output data, running it
 *  through an identity neural network and computing the the mean squared error,
 *  should obviously be zero. */
//______________________________________________________________________________
template <typename Architecture_t>
auto testIdentity()
    -> typename Architecture_t::Scalar_t
{
   using Scalar_t     = typename Architecture_t::Scalar_t;
   using Net_t        = TNet<Architecture_t>;
   using DataLoader_t = TDataLoader<MatrixInput_t, Architecture_t>;

   TMatrixT<Double_t> X(2000, 100), W(2000, 1);
   randomMatrix(X);
   randomMatrix(W);
   MatrixInput_t input(X, X, W);
   DataLoader_t loader(input, 2000, 20, 100, 100);

   Net_t net(20, 100, ELossFunction::kMeanSquaredError);
   net.AddLayer(100,  EActivationFunction::kIdentity);
   net.AddLayer(100,  EActivationFunction::kIdentity);
   net.Initialize(EInitialization::kIdentity);

   Scalar_t maximumError = 0.0;
   for (auto b : loader) {
       auto inputMatrix  = b.GetInput();
       auto outputMatrix = b.GetOutput();
       auto weightMatrix = b.GetWeights();
       Scalar_t error = net.Loss(inputMatrix, outputMatrix, weightMatrix);
       maximumError = std::max(error, maximumError);
   }

   return maximumError;
}

} // namespace DNN
} // namespace TMVA
