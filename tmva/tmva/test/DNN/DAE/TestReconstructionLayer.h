// @(#)root/tmva/tmva/cnn:$Id$
// Author: Akshay Vashistha

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing the Denoise Layer functionalities that are responsible for        *
 *      propagation.                                                              *
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

#ifndef TMVA_TEST_DNN_TEST_RECONSTRUCTION_LAYER_H
#define TMVA_TEST_DNN_TEST_RECONSTRUCTION_LAYER_H

#include "../Utility.h"

#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DAE/ReconstructionLayer.h"

#include "TMVA/DNN/Functions.h"
#include <iostream>

using namespace TMVA::DNN;
using namespace TMVA::DNN::DAE;

template <typename Architecture> auto testLayer(size_t batchSize, size_t visibleUnits, size_t hiddenUnits)
-> void
{
  //using Scalar_t = typename Architecture::Scalar_t;
   using Matrix_t = typename Architecture::Matrix_t;

   using TReconstructionLayer = TReconstructionLayer<Architecture>;


   Matrix_t weights(hiddenUnits,visibleUnits);
   Matrix_t biases1(visibleUnits,1);
   Matrix_t biases2(visibleUnits,1);
   randomMatrix(weights);
   randomMatrix(biases1);
   randomMatrix(biases2);

   std::vector<Matrix_t> Weights, Biases;
   Weights.emplace_back(weights);
   Biases.emplace_back(biases1);
   Biases.emplace_back(biases2);


   TReconstructionLayer dae(batchSize, visibleUnits, hiddenUnits,0.1, EActivationFunction::kSigmoid,Weights,Biases,0.3,1);


   std::vector<Matrix_t> input, compressedInput;
   for(size_t i=0; i<batchSize; i++)
   {
      input.emplace_back(visibleUnits,1);
      compressedInput.emplace_back(hiddenUnits,1);
   }
   Matrix_t inputMatrix(visibleUnits, 1);
   Matrix_t compressMatrix(hiddenUnits,1);




   for(size_t i=0; i<batchSize; i++)
   {
      randomMatrix(inputMatrix);
      Architecture::Copy(input[i],inputMatrix);
      randomMatrix(compressMatrix);
      Architecture::Copy(compressedInput[i],compressMatrix);
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


   std::cout<<"Compressed Input Batch: "<<std::endl;
   for(size_t i=0; i<(size_t)compressedInput.size(); i++)
   {
      for(size_t j=0; j<(size_t)compressedInput[i].GetNrows(); j++)
      {
         for(size_t k=0; k<(size_t)compressedInput[i].GetNcols(); k++)
         {
            std::cout<<compressedInput[i](j,k)<<"\t";
         }
         std::cout<<std::endl;
      }
   }
   std::cout<<std::endl;


   dae.Forward(compressedInput,false);
   dae.Backward(compressedInput,input, compressedInput,input);


   std::cout<<std::endl;
   dae.Print();


}
#endif
