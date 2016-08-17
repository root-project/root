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
// Train the multi-threaded CPU implementation of DNNs on a random //
// linear mapping. In the linear case the minimization problem is  //
// convex and the gradient descent training should converge to the //
// global minimum.                                                 //
/////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestMinimization.h"

using namespace TMVA::DNN;

template <>
auto testMinimization<TCpu<Double_t>>()
   -> Double_t
{
   using Matrix_t = typename TCpu<Double_t>::Matrix_t;
   using Net_t    = TNet<TCpu<Double_t>>;

   size_t nSamples  = 100000;
   size_t nFeatures = 20;
   size_t batchSize = 1024;

   TMatrixT<Double_t> XTrain(nSamples, nFeatures), YTrain(nSamples, 1),
    XTest(batchSize, nFeatures), YTest(batchSize, 1), W(1, nFeatures);

   randomMatrix(W);
   randomMatrix(XTrain);
   randomMatrix(XTest);
   YTrain.MultT(XTrain, W);
   YTest.MultT(XTest, W);

   Net_t net(batchSize, nFeatures, ELossFunction::MEANSQUAREDERROR);
   net.AddLayer(256, EActivationFunction::IDENTITY);
   net.AddLayer(256, EActivationFunction::IDENTITY);
   net.AddLayer(256, EActivationFunction::IDENTITY);
   net.AddLayer(256, EActivationFunction::IDENTITY);
   net.AddLayer(1, EActivationFunction::IDENTITY);
   net.Initialize(EInitialization::GAUSS);

   TGradientDescent<TCpu<Double_t>> minimizer(0.00001, 1, 1);
   MatrixInput_t trainingData(XTrain, YTrain);
   MatrixInput_t testData(XTest, YTest);
   minimizer.TrainTBB(trainingData, nSamples, testData, batchSize, net, 4);

   return 0.0;
}

int main()
{
    testMinimization<TCpu<double, false>>();
}
