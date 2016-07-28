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
// Use the generic tests defined in TestMinimization.h to test the //
// training of Neural Networks for CUDA architectures.             //
/////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMVA/DNN/Minimizers.h"
#include "TestMinimization.h"

using namespace TMVA::DNN;

int main()
{
   using Matrix_t = TMatrixT<Double_t>;
   using Net_t    = TNet<TCuda>;

   Matrix_t XTrain(100000,20), YTrain(100000,20), XTest(20,20), YTest(20,20), W(20, 20);

   randomMatrix(W);

   randomMatrix(XTrain);
   randomMatrix(XTest);

   TReference<Double_t>::MultiplyTranspose(YTrain, XTrain, W);
   TReference<Double_t>::MultiplyTranspose(YTest, XTest, W);

   MatrixInput_t trainData(XTrain, YTrain);
   MatrixInput_t testData(XTest, YTest);

   Net_t net(1000, 20, ELossFunction::MEANSQUAREDERROR);

   net.AddLayer(200, EActivationFunction::IDENTITY);
   net.AddLayer(200, EActivationFunction::IDENTITY);
   net.AddLayer(200, EActivationFunction::IDENTITY);
   net.AddLayer(20, EActivationFunction::IDENTITY);
   net.Initialize(EInitialization::GAUSS);
   auto testnet = net.CreateClone(20);

   TGradientDescent<TCuda> minimizer(0.001, 20, 20);
   minimizer.Train(trainData, 100000, testData, 20, net);

   TMatrixT<Double_t> I(20,20); identityMatrix(I);
   TCudaMatrix ICuda(I);

   testnet.Forward(ICuda);

   TMatrixT<Double_t> WT(20, 20);
   WT.Transpose(W);

   auto error = maximumRelativeError((TMatrixT<Double_t>) testnet.GetOutput(), WT);
   std::cout << "Maximum relative error: " << error << std::endl;
}
