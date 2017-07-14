// @(#)root/tmva $Id$
// Author: Akshay Vashistha (ajatgd)

/*************************************************************************
 * Copyright (C) 2017, ajatgd
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Testing the Encode function                                    //
////////////////////////////////////////////////////////////////////

#include <cmath>
#include <iostream>

#include "TMVA/DNN/Architectures/Reference.h"
#include "TMVA/DNN/Functions.h"
#include "TestDenoiseAE.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::DAE;
using Matrix_t = typename TReference<double>::Matrix_t;

void test1() {
  double inputVector[][10] =
      /* {
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 116, 125, 171, 255, 255,
         150, 93, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 169, 253, 253, 253, 253, 253, 253, 218, 30,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 169, 253, 253, 253, 213, 142, 176, 253, 253, 122, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 52,
           250, 253, 210, 32, 12, 0, 6, 206, 253, 140, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 77, 251, 210, 25, 0,
           0, 0, 122, 248, 253, 65, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 31, 18, 0, 0, 0, 0, 209, 253, 253, 65,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         117, 247, 253, 198, 10, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 76, 247, 253, 231, 63, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 128, 253, 253, 144, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 176, 246,
           253, 159, 12, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 25, 234, 253, 233, 35, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 198, 253, 253,
         141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 78, 248, 253, 189, 12, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           19, 200, 253, 253, 141, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 134, 253, 253, 173, 12,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         248, 253, 253, 25, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 253, 253, 43, 20, 20,
         20, 20, 5, 0, 5, 20, 20, 37, 150, 150, 150, 147,
           10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 248, 253, 253, 253, 253, 253, 253,
         253, 168, 143, 166, 253, 253, 253, 253, 253, 253,
           253, 123, 0, 0, 0, 0, 0, 0, 0, 0, 0, 174, 253, 253, 253, 253, 253,
         253, 253, 253, 253, 253, 253, 249, 247, 247, 169,
           117, 117, 57, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 118, 123, 123, 123, 166,
         253, 253, 253, 155, 123, 123, 41, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0},
          {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 38, 254, 109, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 87, 252, 82, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 135, 241, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 45, 244, 150, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         84, 254, 63, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 202, 223, 11, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 32, 254, 216, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           95, 254, 195, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 140, 254, 77, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 57,
         237, 205, 8, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 124, 255, 165, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 171, 254, 81, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 24,
           232, 215, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 120, 254, 159, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 151,
         254, 142, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 228, 254, 66, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 61, 251, 254, 66, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           141, 254, 205, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 10, 215, 254, 121, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         5, 198, 176, 10, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
         0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
           0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}
         };*/
      {{1, 0, 1, 0, 1, 0, 1, 0, 1, 0}, {0, 1, 0, 1, 0, 1, 0, 1, 0, 1}};

  size_t totalRowsInput = sizeof(inputVector) / sizeof(inputVector[0]);
  size_t totalColumnsInput = sizeof(inputVector[0]) / sizeof(inputVector[0][0]);
  size_t compressedUnits = 5;
  double corruptionLevel = 0.2;
  double learningRate = 0.1;
  size_t fBatchSize = totalRowsInput;

  std::vector<Matrix_t> input;
  for (size_t i = 0; i < totalRowsInput; i++) {
    input.emplace_back(totalColumnsInput, 1);
  }

  std::vector<Matrix_t> corruptedInput;
  for (size_t i = 0; i < totalRowsInput; i++) {
    corruptedInput.emplace_back(totalColumnsInput, 1);
  }

  std::vector<Matrix_t> compressedInput;
  for (size_t i = 0; i < totalRowsInput; i++) {
    compressedInput.emplace_back(compressedUnits, 1);
  }

  std::vector<Matrix_t> reconstructedInput;
  for (size_t i = 0; i < totalRowsInput; i++) {
    reconstructedInput.emplace_back(totalColumnsInput, 1);
  }

  Matrix_t Weights(compressedUnits, totalColumnsInput);
  randomMatrix(Weights);

  Matrix_t fHBiases(compressedUnits, 1);
  for (size_t i = 0; i < (size_t)fHBiases.GetNrows(); i++) {
    for (size_t j = 0; j < (size_t)fHBiases.GetNcols(); j++) {
      fHBiases(i, j) = 0;
    }
  }

  Matrix_t fVBiases(totalColumnsInput, 1);
  for (size_t i = 0; i < (size_t)fVBiases.GetNrows(); i++) {
    for (size_t j = 0; j < (size_t)fVBiases.GetNcols(); j++) {
      fVBiases(i, j) = 0;
    }
  }

  Matrix_t VBiasError(totalColumnsInput, 1);
  Matrix_t HBiasError(compressedUnits, 1);

  /* std::cout<<"Weights";
   for (size_t i=0;i<(size_t)Weights.GetNrows();i++)
    {
      for(size_t j=0; j<(size_t)Weights.GetNcols();j++)
      {
        std::cout<<Weights(i,j)<<std::endl;
      }
      std::cout<<std::endl;
    }*/
  //  std::cout<<"no of matrices"<<input.size()<<std::endl;
  //  std::cout<<"rows"<<input[0].GetNrows()<<"cols"<<input[0].GetNcols()<<std::endl;

  for (size_t i = 0; i < totalRowsInput; i++) {
    for (size_t j = 0; j < (size_t)input[i].GetNrows(); j++) {
      for (size_t k = 0; k < (size_t)input[i].GetNcols(); k++) {
        input[i](j, k) = inputVector[i][j];
      }
    }
    /*for (size_t j = 0; j< (size_t) input[i].GetNrows(); j++)
    {
      for(size_t k=0; k<(size_t)input[i].GetNcols(); k++)
      {
        std::cout<<input[i](j,k)<<std::endl;
      }
    }*/

    std::cout << "corrupting input" << std::endl;
    testCorruptInput<TReference<double>>(input[i], corruptedInput[i],
                                         corruptionLevel);

    std::cout << "compressedInput" << std::endl;
    testEncodeInput<TReference<double>>(corruptedInput[i], compressedInput[i],
                                        Weights);

    std::cout << "reconstructedInput" << std::endl;
    testReconstructInput<TReference<double>>(compressedInput[i],
                                             reconstructedInput[i], Weights);

    std::cout << "Updating weights" << std::endl;
    testUpdateParameter<TReference<double>>(
        input[i], corruptedInput[i], compressedInput[i], reconstructedInput[i],
        fVBiases, fHBiases, Weights, VBiasError, HBiasError, learningRate,
        fBatchSize);
  }
}
int main() {
  std::cout << "Testing for propagation" << std::endl;
  test1();
}
