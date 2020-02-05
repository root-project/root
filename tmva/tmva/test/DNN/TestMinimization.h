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
#include "TRandom3.h"
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
   using Scalar_t = typename Architecture::Scalar_t;
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t    = TNet<Architecture>;

   size_t nSamples  = 256;
   size_t nFeatures = 8;
   size_t batchSize = 8;

   TMatrixT<Double_t> XTrain(nSamples, nFeatures), YTrain(nSamples, 1), WTrain(nSamples, 1),
      XTest(batchSize, nFeatures), YTest(batchSize, 1), WTest(nSamples, 1), K(nFeatures, 1);

   TRandom3 rng{7101};

   // Use random K to generate linear mapping.
   randomMatrix(K, 0., 1., rng);
   randomMatrix(XTrain, 0., 1., rng);
   randomMatrix(XTest, 0., 1., rng);
   YTrain.Mult(XTrain, K);
   YTest.Mult(XTest, K);

   fillMatrix(WTrain, 1.0);
   fillMatrix(WTest, 1.0);

   Net_t net(batchSize, nFeatures, ELossFunction::kMeanSquaredError);
   Architecture::SetRandomSeed(7102);
   net.AddLayer(8, EActivationFunction::kIdentity);
   net.AddLayer(8, EActivationFunction::kIdentity);
   net.AddLayer(1, EActivationFunction::kIdentity);
   net.Initialize(EInitialization::kGauss);

   TGradientDescent<Architecture> minimizer(0.001, 10, 5);
   MatrixInput_t trainingData(XTrain, YTrain, WTrain);
   MatrixInput_t testData(XTest, YTest, WTrain);
   minimizer.Train(trainingData, nSamples, testData, batchSize, net, 1);

   TMatrixT<Double_t> I(nFeatures, nFeatures);
   for (size_t i = 0; i < nFeatures; i++) {
      I(i, i) = 1.0;
   }
   Matrix_t Id(I);
   auto clone = net.CreateClone(nFeatures);
   clone.Forward(Id);
   TMatrixT<Double_t> Y(TMatrixT<Scalar_t>(clone.GetOutput()));

   return maximumRelativeError(Y, K);
}

/** Train a linear neural network on data from two randomly generated linear mappings
 *  from a 8-dimensional input space to a 1-dimensional output space. Set weights
 *  corresponding to the second mapping to zero so that the neural network is forced to
 *  learn the first mapping.
 *  Returns the error of the response of the network to the input containing
 *  only ones to the 1x8 matrix used to generate the training data.
 */
template <typename Architecture>
auto testMinimizationWeights() -> typename Architecture::Scalar_t
{
   using Scalar_t = typename Architecture::Scalar_t;
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t = TNet<Architecture>;

   size_t nSamples  = 256;
   size_t nFeatures = 8;
   size_t batchSize = 8;

   TMatrixT<Double_t> X1(nSamples, nFeatures), X2(nSamples, nFeatures), XTrain(2 * nSamples, nFeatures),
      Y1(nSamples, 1), Y2(nSamples, 1), YTrain(2 * nSamples, 1), W1(nSamples, 1), W2(nSamples, 1), W(2 * nSamples, 1),
      XTest(batchSize, nFeatures), YTest(batchSize, 1), WTest(batchSize, 1), K1(nFeatures, 1), K2(nFeatures, 1);

   TRandom3 rng{7101};

   // Training data from two different linear mappings.
   randomMatrix(K1, 0., 1., rng);
   randomMatrix(K2, 0., 1., rng);
   randomMatrix(X1, 0., 1., rng);
   randomMatrix(X2, 0., 1., rng);
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
   randomMatrix(XTest, 0., 1., rng);
   YTest.Mult(XTest, K2);
   WTest = 1.0;

   Net_t net(batchSize, nFeatures, ELossFunction::kMeanSquaredError);
   Architecture::SetRandomSeed(7102);
   net.AddLayer(8, EActivationFunction::kIdentity);
   net.AddLayer(8, EActivationFunction::kIdentity);
   net.AddLayer(1, EActivationFunction::kIdentity);
   net.Initialize(EInitialization::kGauss);

   TGradientDescent<Architecture> minimizer(0.001, 20, 5);
   MatrixInput_t trainingData(XTrain, YTrain, W);
   MatrixInput_t testData(XTest, YTest, WTest);
   minimizer.TrainMomentum(trainingData, 2 * nSamples, testData, batchSize, net, 0.9, 1);

   TMatrixT<Double_t> I(nFeatures, nFeatures);
   for (size_t i = 0; i < nFeatures; i++) {
      I(i, i) = 1.0;
   }
   Matrix_t Id(I);
   auto clone = net.CreateClone(nFeatures);
   clone.Forward(Id);
   TMatrixT<Double_t> Y(TMatrixT<Scalar_t>(clone.GetOutput()));

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

   size_t nSamples  = 256;
   size_t nFeatures = 8;
   size_t batchSize = 8;

   TMatrixT<Double_t> XTrain(nSamples, nFeatures), YTrain(nSamples, 1), WTrain(nSamples, 1),
      XTest(batchSize, nFeatures), YTest(batchSize, 1), WTest(nSamples, 1), W(nFeatures, 1);

   TRandom3 rng{7101};

   randomMatrix(W, 0., 1., rng);
   randomMatrix(XTrain, 0., 1., rng);
   randomMatrix(XTest, 0., 1., rng);
   YTrain.Mult(XTrain, W);
   YTest.Mult(XTest, W);

   fillMatrix(WTrain, 1.0);
   fillMatrix(WTest, 1.0);

   auto ur = [&rng](Scalar_t /*x*/) {
      return rng.Uniform();
   };

   applyMatrix(WTrain, ur);
   applyMatrix(WTest, ur);

   Net_t net(batchSize, nFeatures, ELossFunction::kMeanSquaredError);
   Architecture::SetRandomSeed(7102);
   net.AddLayer(8, EActivationFunction::kIdentity);
   net.AddLayer(8, EActivationFunction::kIdentity);
   net.AddLayer(1, EActivationFunction::kIdentity);
   net.Initialize(EInitialization::kGauss);

   TGradientDescent<Architecture> minimizer(0.001, 20, 5);
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
   TMatrixT<Double_t> Y(TMatrixT<Scalar_t>(clone.GetOutput()));

   return maximumRelativeError(Y, W);
}
