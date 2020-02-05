// @(#)root/tmva $Id$
// Author: Saurav Shekhar

/*************************************************************************
 * Copyright (C) 2017, Saurav Shekhar                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Generic tests of the RNNLayer Forward pass                     //
////////////////////////////////////////////////////////////////////

#ifndef TMVA_TEST_DNN_TEST_RNN_TEST_FULL
#define TMVA_TEST_DNN_TEST_RNN_TEST_FULL

#include <iostream>
#include <vector>

#include "../Utility.h"

#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DeepNet.h"
#include "TMVA/DNN/Net.h"
#include "TMVA/ROCCurve.h"
#include "TFile.h"
#include "TH1.h"
#include "TGraph.h"
#include "TRandom3.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;

template <typename Architecture>
auto printTensor1(const std::vector<typename Architecture::Matrix_t> &A, const std::string name = "matrix")
-> void
{
  std::cout << name << "\n";
  for (size_t l = 0; l < A.size(); ++l) {
      for (size_t i = 0; i < A[l].GetNrows(); ++i) {
        for (size_t j = 0; j < A[l].GetNcols(); ++j) {
            std::cout << A[l](i, j) << " ";
        }
        std::cout << "\n";
      }
      std::cout << "********\n";
  } 
}

template <typename Architecture>
auto printMatrix1(const typename Architecture::Matrix_t &A, const std::string name = "matrix")
-> void
{
  std::cout << name << "\n";
  for (size_t i = 0; i < A.GetNrows(); ++i) {
    for (size_t j = 0; j < A.GetNcols(); ++j) {
        std::cout << A(i, j) << " ";
    }
    std::cout << "\n";
  }
  std::cout << "********\n";
}

/* Generate a full recurrent neural net
 * like a word generative model */
//______________________________________________________________________________
template <typename Architecture>
auto testFullRNN(size_t batchSize, size_t stateSize, 
                 size_t inputSize, size_t outputSize)
-> void
{
   using Matrix_t   = typename Architecture::Matrix_t;
   using Tensor_t   = typename Architecture::Tensor_t;
   
   // using RNNLayer_t = TBasicRNNLayer<Architecture>;
   // using FCLayer_t  = TDenseLayer<Architecture>;
   // using Reshape_t  = TReshapeLayer<Architecture>;
   using Net_t      = TDeepNet<Architecture>;
   using Scalar_t   = typename Architecture::Scalar_t; 
 
   // check, denselayer takes only first one as input, 
   // so make sure time = 1, in the current case
   size_t timeSteps = 1;
   
   Tensor_t XArch(batchSize, timeSteps, inputSize); // B x T x D
   
   randomBatch(XArch); 
   Tensor_t XRef = XArch; 

     // B x T x D
   //TMatrixT<Double_t> YRef(batchSize, outputSize);    // B x O  (D = O)
   
   Matrix_t YArch(batchSize, outputSize);             // B x O  (D = O)

   std::cerr << "Copying output into input\n";
   for (size_t i = 0; i < batchSize; ++i) {
      for (size_t j = 0; j < outputSize; ++j) {
         YArch(i, j) = XArch(i, 0, j);  // time steps is 1 
      }
   }

   Net_t rnn(batchSize, batchSize, timeSteps, inputSize, 0, 0, 0, ELossFunction::kMeanSquaredError, EInitialization::kGauss);
   //    RNNLayer_t* layer = rnn.AddBasicRNNLayer(stateSize, inputSize, timeSteps, false);
   //    Reshape_t* reshape = rnn.AddReshapeLayer(1, 1, stateSize, true);
   //    FCLayer_t* classifier = rnn.AddDenseLayer(outputSize, EActivationFunction::kIdentity);
   rnn.AddBasicRNNLayer(stateSize, inputSize, timeSteps, false);
   rnn.AddReshapeLayer(1, 1, stateSize, true);
   rnn.AddDenseLayer(outputSize, EActivationFunction::kIdentity);

   Matrix_t W(batchSize, 1);
   for (size_t i = 0; i < batchSize; ++i) W(i, 0) = 1.0;
   rnn.Initialize();

   size_t iter = 0;
   while (iter++ < 50) {
      rnn.Forward(XArch);
      Scalar_t loss = rnn.Loss(YArch, W, false);

      //if (iter % 20 == 0) {
         //for (size_t i = 0; i < inputSize; ++i) std::cout << XRef[0](0, i) << " "; std::cout << "\n";
         //for (size_t i = 0; i < inputSize; ++i) std::cout << rnn.GetLayers().back()->GetOutputAt(0)(0, i) << " "; std::cout << "\n";
      //}
      std::cout << "iteration : " << iter << "  loss: " << loss << std::endl;

      rnn.Backward(XArch, YArch, W);

      rnn.Update(0.1);

   }
}

/* Generate a full recurrent neural net
   with several time steps and using a dense layer afterwards 
   The time steps is fixed 
*/
//______________________________________________________________________________
template <typename Architecture>
auto testFullRNN2(size_t batchSize, size_t stateSize, 
                  size_t inputSize, int seed)
-> void
{
   using Matrix_t   = typename Architecture::Matrix_t;
   using Tensor_t   = typename Architecture::Tensor_t;
   // using RNNLayer_t = TBasicRNNLayer<Architecture>;
   // using FCLayer_t  = TDenseLayer<Architecture>;
   // using Reshape_t  = TReshapeLayer<Architecture>;
   using Net_t      = TDeepNet<Architecture>;
   using Scalar_t   = typename Architecture::Scalar_t;

   bool saveResult = false; 

   TRandom3 rndm(seed); 
 
   // check, denselayer takes only first one as input, 
   // so make sure time = 1, in the current case
   size_t timeSteps = 5;
  
   Tensor_t XRef(batchSize, timeSteps, inputSize); // B x T x D

  ///std::vector<TMatrixT<Double_t>> XRef(batchSize, TMatrixT<Double_t>(timeSteps, inputSize));    // B x T x D
   //TMatrixT<Double_t> YRef(batchSize, outputSize);    // B x O  (D = O)
   //Tensor_t XArch;
   Matrix_t YArch(batchSize, 1);             // B x O  (D = O)
   for (size_t i = 0; i < batchSize; ++i) {
      // provide input data and labels Yarch
      // define if events is class  0 or 1
      int label = rndm.Integer(2);
      YArch(i, 0) = label;
      for (size_t l = 0; l < timeSteps; ++l) {
         double mu = (label == 0) ? 4 : 2*l;
         for (size_t m = 0; m < inputSize; ++m) {
            mu += m;   // shift the varouous inputs
            XRef(i,l,m) = rndm.Gaus( mu, 1);
         }
      }
      //std::cerr << "Copying output into input\n";
   }
   Tensor_t XArch = XRef; 

   bool useRegularization = false;
   double weightDecay = (useRegularization) ? 1. : 0; 
   //Net_t rnn(batchSize, batchSize, timeSteps, inputSize, 0, 0, 0, ELossFunction::kCrossEntropy, EInitialization::kGauss);
   Net_t rnn(batchSize, batchSize, timeSteps, inputSize, 0, 0, 0, ELossFunction::kCrossEntropy, EInitialization::kGauss, ERegularization::kL2, weightDecay);
   //    RNNLayer_t* layer = rnn.AddBasicRNNLayer(stateSize, inputSize, timeSteps, false);
   //    Reshape_t* reshape = rnn.AddReshapeLayer(1, 1, stateSize, true);
   //    FCLayer_t* classifier = rnn.AddDenseLayer(outputSize, EActivationFunction::kIdentity);
   rnn.AddBasicRNNLayer(stateSize, inputSize, timeSteps, false);
   rnn.AddReshapeLayer(1, 1, timeSteps*stateSize, true);
   rnn.AddDenseLayer(10, EActivationFunction::kTanh);
   rnn.AddDenseLayer(1, EActivationFunction::kIdentity);

   Matrix_t W(batchSize, 1);
   for (size_t i = 0; i < batchSize; ++i) W(i, 0) = 1.0;
   rnn.Initialize();

   size_t iter = 0;
   while (iter++ < 50) {
      rnn.Forward(XArch);
      Scalar_t loss = rnn.Loss(YArch, W, useRegularization);

      //if (iter % 20 == 0) {
         //for (size_t i = 0; i < inputSize; ++i) std::cout << XRef[0](0, i) << " "; std::cout << "\n";
         //for (size_t i = 0; i < inputSize; ++i) std::cout << rnn.GetLayers().back()->GetOutputAt(0)(0, i) << " "; std::cout << "\n";
      //}
      std::cout << "iter = " << iter << " loss: " << loss << std::endl;

      rnn.Backward(XArch, YArch, W);

      rnn.Update(0.1);

   }
   // print output
   rnn.Forward(XArch);

   //Matrix_t & out = rnn.GetLayers().back()->GetOutputAt(0);
   //out.Print();
   //YArch.Print();

   // predictions
   Matrix_t yout(batchSize, 1); 
   rnn.Prediction( yout, XArch, EOutputFunction::kSigmoid);

   // predicted output and labels
   TMatrixD result(2, batchSize);
   for (size_t i = 0; i < batchSize; ++i) {
      result(0,i) = YArch(i,0);
      result(1,i) = yout(i,0);
   }
   result.Print();

   auto h0 = new TH1D("h0","h0",50,0,1); 
   auto h1 = new TH1D("h1","h1",50,0,1); 
   // build a roc curve
   std::vector<float> values(batchSize); 
   std::vector<float> targets(batchSize); 
   for (size_t i = 0; i < batchSize; ++i) {
      values[i] = yout(i,0);
      targets[i] =  YArch(i,0);
      if (targets[i] == 0) h0->Fill(values[i] );
      else h1->Fill(values[i] ); 
   }
   TMVA::ROCCurve roc(values, targets);
   std::cout << "ROC integral is " << roc.GetROCIntegral(10) << std::endl;

   if (saveResult)  { 
      TFile * fout = TFile::Open("testRNNResult.root","RECREATE");
      h0->Write();
      h1->Write();
      auto g = roc.GetROCCurve();
      g->Write("roc");
      fout->Close();
   }

   // compute efficiency at 0.5

   double eff1 = h0->Integral(25,50)/h0->Integral(1,50);
   double eff2 = h1->Integral(25,50)/h1->Integral(1,50);

   bool ok = (eff1 > 0.9 && eff2 < 0.1) || (eff1 < 0.1 && eff2 > 0.9);
   if (ok) std::cout << "Test full RNN passed : "; 
   else std::cout << "ERROR : Test full RNN failed : "; 
   std::cout << "Efficiencies are " << eff1 << " and " << eff2 << std::endl;
   
}


#endif
