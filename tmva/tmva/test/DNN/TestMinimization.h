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

   TMatrixT<Double_t> XTrain(4000,20), YTrain(4000,20), XTest(20,20),
       YTest(20,20), W(20, 20);

   randomMatrix(W);
   randomMatrix(XTrain);
   randomMatrix(XTest);
   YTrain.MultT(XTrain, W);
   YTest.MultT(XTest, W);

   Net_t net(20, 20, ELossFunction::MEANSQUAREDERROR);
   net.AddLayer(100, EActivationFunction::IDENTITY);
   net.AddLayer(100, EActivationFunction::IDENTITY);
   net.AddLayer(20, EActivationFunction::IDENTITY);
   net.Initialize(EInitialization::GAUSS);

   TGradientDescent<Architecture> minimizer(0.00001, 5, 10);
   MatrixInput_t trainingData(XTrain, YTrain);
   MatrixInput_t testData(XTest, YTest);
   minimizer.Train(trainingData, 4000, testData, 20, net, 1);

   Matrix_t I(20,20); identityMatrix(I);

   net.Forward(I);

   TMatrixT<Double_t> WT(20, 20);
   WT.Transpose(W);

   auto error = maximumRelativeError((TMatrixT<Double_t>) net.GetOutput(), WT);
   std::cout << "Maximum relative error: " << error << std::endl;

   return error;
}

