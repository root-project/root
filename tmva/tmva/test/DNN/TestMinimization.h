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
 *  from an 8-dimensional input space to a 1-dimensional output space.
 *  Returns the error of the response of the network to the input containing
 *  only ones to the 1x8 matrix used to generate the training data.
 */
template <typename Architecture>
   auto testMinimization()
   -> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t    = TNet<Architecture>;

   size_t nSamples  = 1024;
   size_t nFeatures = 8;
   size_t batchSize = 8;

   TMatrixT<Double_t> XTrain(nSamples, nFeatures), YTrain(nSamples, 1), WTrain(nSamples, 1),
      XTest(batchSize, nFeatures), YTest(batchSize, 1), WTest(nSamples, 1), K(nFeatures, 1);

   // Use random K to generate linear mapping.
   randomMatrix(K);
   randomMatrix(XTrain);
   randomMatrix(XTest);
   YTrain.Mult(XTrain, K);
   YTest.Mult(XTest, K);

   fillMatrix(WTrain, 1.0);
   fillMatrix(WTest, 1.0);

   Net_t net(batchSize, nFeatures, ELossFunction::kMeanSquaredError);
   net.AddLayer(64, EActivationFunction::kIdentity);
   net.AddLayer(64, EActivationFunction::kIdentity);
   net.AddLayer(64, EActivationFunction::kIdentity);
   net.AddLayer(1, EActivationFunction::kIdentity);
   net.Initialize(EInitialization::kGauss);

   TGradientDescent<Architecture> minimizer(0.0001, 5, 1);
   MatrixInput_t trainingData(XTrain, YTrain, WTrain);
   MatrixInput_t testData(XTest, YTest, WTrain);
   minimizer.TrainMomentum(trainingData, nSamples, testData, batchSize, net, 0.8, 1);

   TMatrixT<Double_t> I(nFeatures, nFeatures);
   for (size_t i = 0; i < nFeatures; i++) {
      I(i, i) = 1.0;
   }
   Matrix_t Id(I);
   auto clone = net.CreateClone(nFeatures);
   clone.Forward(Id);
   TMatrixT<Double_t> Y(clone.GetOutput());

   return maximumRelativeError(Y, K);
}

/** Train a linear neural network on data from two randomly generated linear mappings
 *  from a 20-dimensional input space to a 1-dimensional output space. Set weights
 *  corresponding to the second mapping to zero so that the neural network is forced to
 *  learn the first mapping.
 *  Returns the error of the response of the network to the input containing
 *  only ones to the 1x20 matrix used to generate the training data.
 */
template <typename Architecture>
auto testMinimizationWeights() -> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t = TNet<Architecture>;

   size_t nSamples = 10000;
   size_t nFeatures = 20;
   size_t batchSize = 256;

   TMatrixT<Double_t> X1(nSamples, nFeatures), X2(nSamples, nFeatures), XTrain(2 * nSamples, nFeatures),
      Y1(nSamples, 1), Y2(nSamples, 1), YTrain(2 * nSamples, 1), W1(nSamples, 1), W2(nSamples, 1), W(2 * nSamples, 1),
      XTest(batchSize, nFeatures), YTest(batchSize, 1), WTest(batchSize, 1), K1(nFeatures, 1), K2(nFeatures, 1);

   // Training data from two different linear mappings.
   randomMatrix(K1);
   randomMatrix(K2);
   randomMatrix(X1);
   randomMatrix(X2);
   Y1.Mult(X1, K1);
   Y2.Mult(X2, K2);
   XTrain.SetSub(0, 0, X1);
   XTrain.SetSub(nSamples, 0, X2);
   YTrain.SetSub(0, 0, Y1);
   YTrain.SetSub(nSamples, 0, Y2);

   W1 = 0.0;
   W2 = 1.0;
   W.SetSub(0, 0, W1);
   W.SetSub(nSamples, 0, W2);

   // Test data from only the first mapping;
   randomMatrix(XTest);
   YTest.Mult(XTest, K2);
   WTest = 1.0;

   Net_t net(batchSize, nFeatures, ELossFunction::kMeanSquaredError);
   net.AddLayer(64, EActivationFunction::kIdentity);
   net.AddLayer(64, EActivationFunction::kIdentity);
   net.AddLayer(64, EActivationFunction::kIdentity);
   net.AddLayer(1, EActivationFunction::kIdentity);
   net.Initialize(EInitialization::kGauss);

   TGradientDescent<Architecture> minimizer(0.0001, 5, 1);
   MatrixInput_t trainingData(XTrain, YTrain, W);
   MatrixInput_t testData(XTest, YTest, WTest);
   minimizer.TrainMomentum(trainingData, 2 * nSamples, testData, batchSize, net, 0.8, 1);

   TMatrixT<Double_t> I(nFeatures, nFeatures);
   for (size_t i = 0; i < nFeatures; i++) {
      I(i, i) = 1.0;
   }
   Matrix_t Id(I);
   auto clone = net.CreateClone(nFeatures);
   clone.Forward(Id);
   TMatrixT<Double_t> Y(clone.GetOutput());

   return maximumRelativeError(Y, K2);
}

/** Similar to testMinimization() as the function above except that
 *  it uses momentum for the training */
template <typename Architecture>
   auto testMinimizationMomentum()
   -> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   using Scalar_t = typename Architecture::Scalar_t;
   using Net_t    = TNet<Architecture>;

   size_t nSamples  = 10000;
   size_t nFeatures = 20;
   size_t batchSize = 256;

   TMatrixT<Double_t> XTrain(nSamples, nFeatures), YTrain(nSamples, 1), WTrain(nSamples, 1),
      XTest(batchSize, nFeatures), YTest(batchSize, 1), WTest(nSamples, 1), W(nFeatures, 1);

   randomMatrix(W);
   randomMatrix(XTrain);
   randomMatrix(XTest);
   YTrain.Mult(XTrain, W);
   YTest.Mult(XTest, W);

   fillMatrix(WTrain, 1.0);
   fillMatrix(WTest, 1.0);

   auto ur = [](Scalar_t /*x*/) {
      TRandom rand(clock());
      return rand.Uniform();
   };

   applyMatrix(WTrain, ur);
   applyMatrix(WTest, ur);

   Net_t net(batchSize, nFeatures, ELossFunction::kMeanSquaredError);
   net.AddLayer(64, EActivationFunction::kIdentity);
   net.AddLayer(64, EActivationFunction::kIdentity);
   net.AddLayer(64, EActivationFunction::kIdentity);
   net.AddLayer(1, EActivationFunction::kIdentity);
   net.Initialize(EInitialization::kGauss);

   TGradientDescent<Architecture> minimizer(0.0001, 5, 5);
   MatrixInput_t trainingData(XTrain, YTrain, WTrain);
   MatrixInput_t testData(XTest, YTest, WTest);
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
