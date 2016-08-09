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

template <typename Architecture>
   auto testMinimization()
   -> typename Architecture::Scalar_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t    = TNet<Architecture>;
  
   size_t nSamples  = 100000;
   size_t nFeatures = 20;
   size_t batchSize = 1000;

   TMatrixT<Double_t> XTrain(nSamples, nFeatures), YTrain(nSamples, 1),
    XTest(batchSize, nFeatures), YTest(batchSize, 1), W(1, nFeatures);

   randomMatrix(W);
   randomMatrix(XTrain);
   randomMatrix(XTest);
   YTrain.MultT(XTrain, W);
   YTest.MultT(XTest, W);

   Net_t net(batchSize, nFeatures, ELossFunction::MEANSQUAREDERROR);
   net.AddLayer(1000, EActivationFunction::TANH);
   net.AddLayer(1000, EActivationFunction::TANH);
   net.AddLayer(1000, EActivationFunction::TANH);
   net.AddLayer(1, EActivationFunction::IDENTITY);
   net.Initialize(EInitialization::GAUSS);

   TGradientDescent<Architecture> minimizer(0.000001, 1, 10);
   MatrixInput_t trainingData(XTrain, YTrain);
   MatrixInput_t testData(XTest, YTest);
   minimizer.Train(trainingData, nSamples, testData, batchSize, net, 4);

   return 0.0;
}

