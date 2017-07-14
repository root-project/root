// @(#)root/tmva $Id$
// Author: Akshay Vashistha (ajatgd)

/*************************************************************************
 * Copyright (C) 2016, ajatgd
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#include "../Utility.h"
#include "TMVA/DNN/DAE/DenoiseAE.h"
#include "TMVA/DNN/Functions.h"
#include <iostream>

using namespace TMVA::DNN;
using namespace TMVA::DNN::DAE;

template <typename Architecture> auto testLayer() -> void {
  using Scalar_t = typename Architecture::Scalar_t;
  using Matrix_t = typename Architecture::Matrix_t;
  using TDAE = TDAE<Architecture>;

  TDAE dae(1, 10, 5);
  dae.Initialize(EInitialization::kUniform);

  Matrix_t input(10, 1);
  randomMatrix(input);
  Matrix_t corruptedInput(10, 1);
  Matrix_t compressedInput(5, 1);
  Matrix_t reconstructedInput(10, 1);
  Scalar_t corruptionLevel = 0.2;
  dae.Corruption(input, corruptedInput, corruptionLevel);
  dae.Encoding(input, compressedInput);
  dae.Reconstruction(compressedInput, reconstructedInput);

  std::cout << "input" << std::endl;
  for (size_t i = 0; i < (size_t)input.GetNrows(); i++) {
    for (size_t j = 0; j < (size_t)input.GetNcols(); j++) {
      std::cout << input(i, j) << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << "corrupted" << std::endl;
  for (size_t i = 0; i < (size_t)corruptedInput.GetNrows(); i++) {
    for (size_t j = 0; j < (size_t)corruptedInput.GetNcols(); j++) {
      std::cout << corruptedInput(i, j) << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << "compressed" << std::endl;
  for (size_t i = 0; i < (size_t)compressedInput.GetNrows(); i++) {
    for (size_t j = 0; j < (size_t)compressedInput.GetNcols(); j++) {
      std::cout << compressedInput(i, j) << "\t";
    }
    std::cout << std::endl;
  }
  std::cout << "reconstructed" << std::endl;
  for (size_t i = 0; i < (size_t)reconstructedInput.GetNrows(); i++) {
    for (size_t j = 0; j < (size_t)reconstructedInput.GetNcols(); j++) {
      std::cout << reconstructedInput(i, j) << "\t";
    }
    std::cout << std::endl;
  }
}
//_____________________________________________________________________________
template <typename Architecture> auto testTraining() -> void {
  using Scalar_t = typename Architecture::Scalar_t;
  using Matrix_t = typename Architecture::Matrix_t;
  using TDAE = TDAE<Architecture>;

  TDAE dae(1, 10, 5);
  dae.Initialize(EInitialization::kUniform);
  Matrix_t input(10, 1);
  randomMatrix(input);
  Scalar_t corruptionLevel = 0.2;
  Scalar_t learningRate = 0.1;
  dae.TrainLayer(input, learningRate, corruptionLevel);
}
