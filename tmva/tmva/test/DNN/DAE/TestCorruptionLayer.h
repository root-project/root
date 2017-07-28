// @(#)root/tmva/tmva/cnn:$Id$
// Author: Akshay Vashistha

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing the Corruption Layer.                                                              *
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

#ifndef TMVA_TEST_DNN_TEST_CORRUPTION_LAYER_H
#define TMVA_TEST_DNN_TEST_CORRUPTION_LAYER_H

#include "../Utility.h"

#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DAE/CorruptionLayer.h"

#include "TMVA/DNN/Functions.h"
#include <iostream>

using namespace TMVA::DNN;
using namespace TMVA::DNN::DAE;

template <typename Architecture> auto testLayer(size_t batchSize, size_t visibleUnits)
-> void
{
   //using Scalar_t = typename Architecture::Scalar_t;
   using Matrix_t = typename Architecture::Matrix_t;

   using TCorruptionLayer = TCorruptionLayer<Architecture>;

   TCorruptionLayer dae(batchSize, visibleUnits, 1, 0.3);

   std::vector<Matrix_t> input, corruptedInput;
   for(size_t i=0; i<batchSize; i++)
   {
      input.emplace_back(visibleUnits,1);
      corruptedInput.emplace_back(visibleUnits,1);
   }
   Matrix_t inputMatrix(visibleUnits, 1);

   for(size_t i=0; i<batchSize; i++)
   {
      randomMatrix(inputMatrix);
      Architecture::Copy(input[i],inputMatrix);
   }
   std::cout<<"Input Batch: "<<std::endl;
   for(size_t i=0; i<batchSize; i++)
   {
      for(size_t j=0; j<(size_t)input[i].GetNrows(); j++)
      {
         for(size_t k=0; k<(size_t)input[i].GetNcols();k++)
         {
            std::cout<<input[i](j,k)<<"\t";
         }
         std::cout<<std::endl;
      }
   }
   std::cout<<std::endl;

   dae.Forward(input,false);

   std::cout<<std::endl;
   std::cout<<std::endl;
   dae.Print();


}
#endif
