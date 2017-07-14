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
#include "TMVA/DNN/DAE/StackedNet.h"
#include "TMVA/DNN/Functions.h"
#include <iostream>

using namespace TMVA::DNN;
using namespace TMVA::DNN::DAE;

template <typename Architecture> auto testLogistic() -> void {
  using Scalar_t = typename Architecture::Scalar_t;
  using Matrix_t = typename Architecture::Matrix_t;
  using Logistic = LogisticRegressionLayer<Architecture>;
  Scalar_t learningRate = 0.1;
  Logistic logistic(1, 6, 2);
  logistic.Initialize();
  Matrix_t input(6, 1);
  randomMatrix(input);
  Matrix_t output(2, 1);
  randomMatrix(output);
  logistic.TrainLogReg(input, output, learningRate);
  logistic.PredictLogReg(input, output, learningRate);

  std::cout << "input" << std::endl;
  for (size_t i = 0; i < (size_t)input.GetNrows(); i++) {
    for (size_t j = 0; j < (size_t)input.GetNcols(); j++) {
      std::cout << input(i, j) << "\t";
    }
    std::cout << std::endl;
  }

  std::cout << "output" << std::endl;
  for (size_t i = 0; i < (size_t)output.GetNrows(); i++) {
    for (size_t j = 0; j < (size_t)output.GetNcols(); j++) {
      std::cout << output(i, j) << "\t";
    }
    std::cout << std::endl;
  }
}

template <typename Architecture> auto testTransform() -> void {
  using Scalar_t = typename Architecture::Scalar_t;
  using Matrix_t = typename Architecture::Matrix_t;
  using Transform = TransformLayer<Architecture>;

  Transform trans(1, 10, 5);
  Matrix_t weights(5, 10);
  randomMatrix(weights);
  Matrix_t biases(5, 1);
  randomMatrix(biases);
  trans.Initialize(weights, biases);
  Matrix_t input(10, 1);
  randomMatrix(input);
  Matrix_t transformed(5, 1);
  trans.Transform(input, transformed);
  std::cout << "input" << std::endl;
  for (size_t i = 0; i < (size_t)input.GetNrows(); i++) {
    for (size_t j = 0; j < (size_t)input.GetNcols(); j++) {
      std::cout << input(i, j) << "\t";
    }
    std::cout << std::endl;
  }

  std::cout << "transformed" << std::endl;
  for (size_t i = 0; i < (size_t)transformed.GetNrows(); i++) {
    for (size_t j = 0; j < (size_t)transformed.GetNcols(); j++) {
      std::cout << transformed(i, j) << "\t";
    }
    std::cout << std::endl;
  }
}
