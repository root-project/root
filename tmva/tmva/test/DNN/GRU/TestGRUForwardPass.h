// @(#)root/tmva $Id$
// Author: Surya S Dwivedi 07/06/2019

/*************************************************************************
 * Copyright (C) 2019, Surya S Dwivedi                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Generic tests of the GRU-Layer Forward Pass                   //
////////////////////////////////////////////////////////////////////

#ifndef TMVA_TEST_DNN_TEST_GRU_TEST_GRU_FWDPASS_H
#define TMVA_TEST_DNN_TEST_GRU_TEST_GRU_FWDPASS_H

#include <iostream>
#include <vector>

#include "../Utility.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DeepNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;

//______________________________________________________________________________
/* Prints out Tensor, printTensor1(A, matrix) */
template <typename Architecture>
auto printTensor1(const typename Architecture::Tensor_t &A, const std::string & name = "matrix")
-> void
{
   std::cout << name << "\n";
   for (size_t l = 0; l < A.GetFirstSize(); ++l) {
      for (size_t i = 0; i < (size_t) A[l].GetNrows(); ++i) {
         for (size_t j = 0; j < (size_t) A[l].GetNcols(); ++j) {
            std::cout << A[l](i, j) << " ";
         }
         std::cout << "\n";
      }
      std::cout << "********\n";
  }
}



//______________________________________________________________________________
/* Prints out Matrix, printMatrix1(A, matrix) */
template <typename Architecture>
auto printMatrix1(const typename Architecture::Matrix_t &A, const std::string name = "matrix")
-> void
{
   std::cout << name << "\n";
   for (size_t i = 0; i < (size_t) A.GetNrows(); ++i) {
      for (size_t j = 0; j < (size_t) A.GetNcols(); ++j) {
         std::cout << A(i, j) << " ";
      }
      std::cout << "\n";
   }
   std::cout << "********\n";
}

double sigmoid(double x) {
   return 1 /( 1 + exp(-x));
}

/*! Generic sample test for forward propagation in GRU network. */
//______________________________________________________________________________
template <typename Architecture>
auto testForwardPass(size_t timeSteps, size_t batchSize, size_t stateSize, size_t inputSize, bool debug = false, double tol = 1.E-5)
-> Bool_t
{
   using Matrix_t = typename Architecture::Matrix_t;
   using Tensor_t = typename Architecture::Tensor_t;
   using GRULayer_t = TBasicGRULayer<Architecture>;
   using Net_t = TDeepNet<Architecture>;


   //______________________________________________________________________________
   /* Input Gate: Numerical example.
    * Reference: https://medium.com/@aidangomez/let-s-do-this-f9b699de31d9
    * TODO: Numerical example for other gates to verify forward pass values and
    * backward pass values. */
   //______________________________________________________________________________

   // Defining inputs.
   Tensor_t XRef;
   Tensor_t XArch, arr_XArch;

   // std::cout << "Testing GRU Forward for  bs = " << batchSize << " timeSteps = " << timeSteps
   //       << " inputSize = " << inputSize << " outputSize = " << stateSize << std::endl;

   XRef = Architecture::CreateTensor( timeSteps,batchSize, inputSize);
   XArch = Architecture::CreateTensor( batchSize, timeSteps,inputSize);
   arr_XArch = Architecture::CreateTensor( timeSteps,batchSize, inputSize);


   for (size_t i = 0; i < timeSteps; ++i) {
      Matrix_t m = XRef[i];
      randomMatrix(m);
   }

   Architecture::Copy(arr_XArch, XRef); // B x T x D
   Architecture::Rearrange( XArch, XRef);  // Copy from XRef to XArch

   Net_t gru(batchSize, batchSize, timeSteps, inputSize, 0, 0, 0, ELossFunction::kMeanSquaredError, EInitialization::kGauss);
   // make a GRU returning the full sequence
   GRULayer_t* layer = gru.AddBasicGRULayer(stateSize, inputSize, timeSteps, false, true);

   layer->Initialize();
   layer->Print();

   /*! unpack weights for each gate. */
   Matrix_t weightsReset = layer->GetWeightsResetGate();         // H x D
   Matrix_t weightsCandidate = layer->GetWeightsCandidate();     // H x D
   Matrix_t weightsUpdate = layer->GetWeightsUpdateGate();       // H x D
   Matrix_t weightsResetState = layer->GetWeightsResetGateState();       // H x H
   Matrix_t weightsCandidateState = layer->GetWeightsCandidateState();   // H x H
   Matrix_t weightsUpdateState = layer->GetWeightsUpdateGateState();     // H x H
   Matrix_t resetBiases = layer->GetResetGateBias();                     // H x 1
   Matrix_t candidateBiases = layer->GetCandidateBias();                 // H x 1
   Matrix_t updateBiases = layer->GetUpdateGateBias();                   // H x 1

   /*! Get previous hidden state and previous cell state. */
   Matrix_t hiddenState(batchSize, stateSize);       // B x H
   Architecture::Copy(hiddenState, layer->GetState());

   /*! Get each gate values. */
   Matrix_t resetGate = layer->GetResetGateValue();            // B x H
   Matrix_t candidateValue = layer->GetCandidateValue();       // B x H
   Matrix_t updateGate = layer->GetUpdateGateValue();          // B x H

   /*! Temporary Matrices. */
   Matrix_t resetTmp(batchSize, stateSize);
   Matrix_t candidateTmp(batchSize, stateSize);
   Matrix_t updateTmp(batchSize, stateSize);
   Matrix_t tmp(batchSize, stateSize);

   gru.Forward(XArch);

   Tensor_t outputArch = layer->GetOutput();

   Tensor_t arr_outputArch = Architecture::CreateTensor( timeSteps,batchSize, stateSize);

   //Architecture::PrintTensor(outputArch,"GRU output");

   Architecture::Rearrange(arr_outputArch, outputArch); // B x T x H

   if (debug)
      Architecture::PrintTensor(arr_outputArch,"GRU FOrward output");

   Double_t maximumError = 0.0;

   Architecture::InitializeZero(hiddenState);

   /*! Element-wise matrix multiplication of previous hidden
    *  state and weights of previous state followed by computing
    *  next hidden state and next cell state. */
   for (size_t t = 0; t < timeSteps; ++t) {

      Architecture::MultiplyTranspose(resetTmp, hiddenState, weightsResetState);
      Architecture::MultiplyTranspose(resetGate, XRef[t], weightsReset);
      Architecture::ScaleAdd(resetGate, resetTmp);

      Architecture::MultiplyTranspose(updateTmp, hiddenState, weightsUpdateState);
      Architecture::MultiplyTranspose(updateGate, XRef[t], weightsUpdate);
      Architecture::ScaleAdd(updateGate, updateTmp);

      Architecture::AddRowWise(resetGate, resetBiases);
      Architecture::AddRowWise(updateGate, updateBiases);

      /*! Apply activation function to each computed gate values. */
      applyMatrix(resetGate, [](double i) { return sigmoid(i); });
      applyMatrix(updateGate, [](double f) { return sigmoid(f); });

      if (debug)
         Architecture::PrintTensor(Tensor_t(resetGate), "reset gate");

      Architecture::Hadamard(resetGate, hiddenState);

      Architecture::MultiplyTranspose(candidateTmp, resetGate, weightsCandidateState);
      Architecture::MultiplyTranspose(candidateValue, XRef[t], weightsCandidate);
      Architecture::ScaleAdd(candidateValue, candidateTmp);

      Architecture::AddRowWise(candidateValue, candidateBiases);
      applyMatrix(candidateValue, [](double c) { return tanh(c); });

      /*! Computing next cell state and next hidden state. */

      if (debug) {
         Architecture::PrintTensor(Tensor_t(updateGate), "update gate");
         Architecture::PrintTensor(Tensor_t(candidateValue), "candidate value");
      }

      Architecture::Copy(tmp, updateGate); // H X 1
      for (size_t j = 0; j < (size_t) tmp.GetNcols(); j++) {
         for (size_t i = 0; i < (size_t) tmp.GetNrows(); i++) {
            tmp(i,j) = 1 - tmp(i,j);
         }
      }

      if (debug) Architecture::PrintTensor(Tensor_t(tmp), "tmp");

      // Update state
      Architecture::Hadamard(updateGate, hiddenState);
      Architecture::Hadamard(candidateValue, tmp);
      Architecture::ScaleAdd(candidateValue, updateGate);
      Architecture::Copy(hiddenState, candidateValue);


      TMatrixT<Double_t> output = arr_outputArch[t];


      if (debug) {
         std::cout << "GRU output at current time " << std::endl;
         output.Print();
         Architecture::PrintTensor(Tensor_t(hiddenState),"expected output at current time");
      }

      Double_t error = maximumRelativeError(output, hiddenState);
      std::cout << "Time " << t << " Delta: " << error << "\n";

      maximumError = std::max(error, maximumError);
   }

   if (maximumError > tol)
      std::cout << "ERROR: - GRU Forward pass test failed !  - Max dev is ";
   else
      std::cout << " Test GRU forward passed ! -   Max dev is ";

   std::cout << maximumError << std::endl;
   return maximumError < tol;
}

#endif // TMVA_TEST_DNN_TEST_RNN_TEST_GRU_FWDPASS_H
