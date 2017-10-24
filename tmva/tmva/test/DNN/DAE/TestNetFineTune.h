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

#ifndef TMVA_TEST_DNN_TEST_DEEPNET_AE_FINETUNE_H
#define TMVA_TEST_DNN_TEST_DEEPNET_AE_FINETUNE_H

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
   using Matrix_t = typename Architecture::Matrix_t;

   double train_X[][6] = {
      {1, 1, 1, 0, 0, 0},
      {1, 0, 1, 0, 0, 0},
      {1, 1, 1, 0, 0, 0},
      {0, 0, 1, 1, 1, 0},
      {0, 0, 1, 1, 0, 0},
      {0, 0, 1, 1, 1, 0}
   };
   double train_Y[][2] = {
      {1, 0},
      {1, 0},
      {1, 0},
      {0, 1},
      {0, 1},
      {0, 1}
   };

   double test_X[][6] = {
      {1, 0, 1, 0, 0, 0},
      {0, 0, 1, 1, 1, 0}
   };

   std::vector<Matrix_t>input, inputLabel, testInput;
   for(size_t i=0; i<6;i++)
   {
      input.emplace_back(6,1);
      inputLabel.emplace_back(2,1);
   }
   for(size_t i=0; i<2;i++)
   {
      testInput.emplace_back(6,1);
   }

   for(size_t i=0; i<6; i++)
   {
      for(size_t j=0; j<(size_t)input[i].GetNrows(); j++)
      {
         for(size_t k=0; k<(size_t)input[i].GetNcols(); k++)
         {
            input[i](j,k)=train_X[i][j];
         }
      }
   }

   for(size_t i=0; i<6; i++)
   {
      for(size_t j=0; j<(size_t)inputLabel[i].GetNrows(); j++)
      {
         for(size_t k=0; k<(size_t)inputLabel[i].GetNcols(); k++)
         {
            inputLabel[i](j,k)=train_Y[i][j];
         }
      }
   }

   for(size_t i=0; i<2; i++)
   {
      for(size_t j=0; j<(size_t)testInput[i].GetNrows(); j++)
      {
         for(size_t k=0; k<(size_t)testInput[i].GetNcols(); k++)
         {
            testInput[i](j,k)=test_X[i][j];
         }
      }
   }


   size_t testDataBatchSize =2;

   net.FineTune(input,
                testInput,
                inputLabel,2,testDataBatchSize, 0.1, 1000);
}

template <typename Architecture> auto testNet()
-> void
{
   using Net_t = TDeepNet<Architecture>;
   size_t batchSize = 6;
   Net_t convNet(batchSize, 1, 1, 1, 1, 1, 1,
                 ELossFunction::kMeanSquaredError, EInitialization::kGauss);

   constructDeepAutoEncoderNet(convNet);
 }
#endif
