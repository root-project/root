// @(#)root/tmva $Id$
// Author: Saurav Shekhar

/*************************************************************************
 * Copyright (C) 2017, Saurav Shekhar
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Generic tests of the RNNLayer Forward pass                     //
////////////////////////////////////////////////////////////////////

#ifndef TMVA_TEST_DNN_TEST_CNN_TEST_CONV_BACKPROPAGATION_H
#define TMVA_TEST_DNN_TEST_CNN_TEST_CONV_BACKPROPAGATION_H

#include <iostream>
#include "../Utility.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/RNN/RNNLayer.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;

/*! Generate a RNNLayer, perform one forward pass */
//______________________________________________________________________________
template <typename Architecture>
auto testForwardPass(size_t batchSize, size_t stateSize, size_t inputSize)
-> typename Architecture::Scalar_t 
{
  using Scalar_t   = typename Architecture::Scalar_t;
  using Matrix_t   = typename Architecture::Matrix_t;
  using RNNLayer_t = TBasicRNNLayer<Architecture>;  
 
  Scalar_t maximum_error = 0.0;

  return maximum_error;
}

#endif
