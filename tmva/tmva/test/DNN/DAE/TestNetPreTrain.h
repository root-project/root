// @(#)root/tmva/tmva/cnn:$Id$
// Author: Akshay Vashistha

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing the AutoEncoder DeepNet .                                         *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Akshay Vashistha    <akshayvashistha1995@gmail.com>  - CERN, Switzerland  *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#ifndef TMVA_TEST_DNN_TEST_DEEPNET_AE_H
#define TMVA_TEST_DNN_TEST_DEEPNET_AE_H

#include "../Utility.h"

#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DeepNet.h"

#include "TMVA/DNN/Functions.h"
#include <iostream>

using namespace TMVA::DNN;
using namespace TMVA::DNN::DAE;

template <typename Architecture> auto constructDeepAutoEncoderNet(TDeepNet<Architecture> &net)
-> void
{
   using Scalar_t = typename Architecture::Scalar_t;
   using Matrix_t = typename Architecture::Matrix_t;

   std::vector<Matrix_t> input;
   size_t batchSize=10;
   size_t visibleUnits=6;
   std::vector<size_t> numHiddenUnitsPerLayer = {4,3};
   Scalar_t learningRate =0.1;
   Scalar_t corruptionLevel = 0.3;
   Scalar_t dropoutProbability = 1;
   size_t epochs=5;

   Matrix_t inputMatrix(visibleUnits,1);
   for(size_t i=0; i<batchSize; i++)
   {
      input.emplace_back(visibleUnits,1);
   }
   for(size_t i=0; i<batchSize; i++)
   {
      randomMatrix(inputMatrix);
      Architecture::Copy(input[i],inputMatrix );
   }
   net.PreTrain(input,
                numHiddenUnitsPerLayer,
                learningRate, corruptionLevel,
                dropoutProbability, epochs,
                EActivationFunction::kSigmoid, false);
}

template <typename Architecture> auto testNet()
-> void
{
   using Scalar_t = typename Architecture::Scalar_t;
   using Matrix_t = typename Architecture::Matrix_t;
   using Net_t = TDeepNet<Architecture>;
   size_t batchSize = 10;
   Net_t convNet(batchSize, 1, 1, 1, 1, 1, 1,
                 ELossFunction::kMeanSquaredError, EInitialization::kGauss);

   constructDeepAutoEncoderNet(convNet);
 }
#endif
