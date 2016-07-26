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
   using Net_t    = TNet<TCuda<false>>;

   Matrix_t XTrain(4000,20), YTrain(4000,20), XTest(20,20), YTest(20,20), W(20, 20);

   randomMatrix(W);

   for (size_t i = 0; i < 4000; i++) {
      for (size_t j = 0; j < 20; j++) {
         XTrain(i,j) = i;
         YTrain(i,j) = i;
      }
   }

   randomMatrix(XTrain);
   randomMatrix(XTest);

   TReference<Double_t>::MultiplyTranspose(YTrain, XTrain, W);
   TReference<Double_t>::MultiplyTranspose(YTest, XTest, W);

   MatrixInput_t trainData(XTrain, YTrain);
   MatrixInput_t testData(XTest, YTest);

   Net_t net(20, 20, ELossFunction::MEANSQUAREDERROR);
   net.AddLayer(100, EActivationFunction::IDENTITY);
   net.AddLayer(20, EActivationFunction::IDENTITY);
   net.Initialize(EInitialization::GAUSS);

   TGradientDescent<TCuda<false>> minimizer(0.001, 1, 20);
   minimizer.Train(trainData, 4000, testData, 20, net);

   TMatrixT<Double_t> I(20,20); identityMatrix(I);
   TCudaMatrix ICuda(I);

   net.Forward(ICuda);

   TMatrixT<Double_t> WT(20, 20);
   WT.Transpose(W);

   auto error = maximumRelativeError((TMatrixT<Double_t>) net.GetOutput(), WT);
   std::cout << "Maximum relative error: " << error << std::endl;
}
