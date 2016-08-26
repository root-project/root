// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/////////////////////////////////////////////////////////////////////
// Test Standard Minimizer                                         //
//                                                                 //
// This test trains a linear neural network on a linear function   //
// F(x) = W * x and computes the relative error between the matrix //
// W' representing the linear function learned by the net to the   //
// orignal matrix W.                                               //
/////////////////////////////////////////////////////////////////////

#include "TMatrix.h"
#include "TMVA/DNN/Minimizers.h"
#include "TMVA/DNN/Net.h"
#include "Utility.h"

using namespace TMVA::DNN;

/** Train a linear neural network on a randomly generated linear mapping
 *  from a 20-dimensional input space to a 1-dimensional output space.
 *  Returns the error of the response of the network to the input containing
 *  only ones to the 1x20 matrix generating the mapping.
 */
template <typename Architecture>
   auto testMinimization()
   -> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t    = TNet<Architecture>;

   size_t nSamples  = 10000;
   size_t nFeatures = 20;
   size_t batchSize = 256;

   TMatrixT<Double_t> XTrain(nSamples, nFeatures), YTrain(nSamples, 1),
   XTest(batchSize, nFeatures), YTest(batchSize, 1), W(nFeatures, 1);

   randomMatrix(W);
   randomMatrix(XTrain);
   randomMatrix(XTest);
   YTrain.Mult(XTrain, W);
   YTest.Mult(XTest, W);

   Net_t net(batchSize, nFeatures, ELossFunction::kMeanSquaredError);
   net.AddLayer(64, EActivationFunction::kIdentity);
   net.AddLayer(64, EActivationFunction::kIdentity);
   net.AddLayer(64, EActivationFunction::kIdentity);
   net.AddLayer(1, EActivationFunction::kIdentity);
   net.Initialize(EInitialization::kGauss);

   TGradientDescent<Architecture> minimizer(0.0001, 5, 1);
   MatrixInput_t trainingData(XTrain, YTrain);
   MatrixInput_t testData(XTest, YTest);
   minimizer.TrainMomentum(trainingData, nSamples, testData, batchSize, net, 0.8, 1);

   TMatrixT<Double_t> I(nFeatures, nFeatures);
   for (size_t i = 0; i < nFeatures; i++) {
      I(i, i) = 1.0;
   }
   Matrix_t Id(I);
   auto clone = net.CreateClone(nFeatures);
   clone.Forward(Id);
   TMatrixT<Double_t> Y(clone.GetOutput());

   return maximumRelativeError(Y, W);
}

/** Similar to testMinimization() as the function above except that
 *  it uses momentum for the training */
template <typename Architecture>
   auto testMinimizationMomentum()
   -> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t    = TNet<Architecture>;

   size_t nSamples  = 10000;
   size_t nFeatures = 20;
   size_t batchSize = 256;

   TMatrixT<Double_t> XTrain(nSamples, nFeatures), YTrain(nSamples, 1),
   XTest(batchSize, nFeatures), YTest(batchSize, 1), W(nFeatures, 1);

   randomMatrix(W);
   randomMatrix(XTrain);
   randomMatrix(XTest);
   YTrain.Mult(XTrain, W);
   YTest.Mult(XTest, W);

   Net_t net(batchSize, nFeatures, ELossFunction::kMeanSquaredError);
   net.AddLayer(64, EActivationFunction::kIdentity);
   net.AddLayer(64, EActivationFunction::kIdentity);
   net.AddLayer(64, EActivationFunction::kIdentity);
   net.AddLayer(1, EActivationFunction::kIdentity);
   net.Initialize(EInitialization::kGauss);

   TGradientDescent<Architecture> minimizer(0.0001, 5, 5);
   MatrixInput_t trainingData(XTrain, YTrain);
   MatrixInput_t testData(XTest, YTest);
   minimizer.TrainMomentum(trainingData, nSamples, testData, batchSize, net, 0.9, 1);

   TMatrixT<Double_t> I(nFeatures, nFeatures);
   for (size_t i = 0; i < nFeatures; i++) {
      I(i, i) = 1.0;
   }
   Matrix_t Id(I);
   auto clone = net.CreateClone(nFeatures);
   clone.Forward(Id);
   TMatrixT<Double_t> Y(clone.GetOutput());

   return maximumRelativeError(Y, W);
}
