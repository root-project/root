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

#ifndef TMVA_TEST_DNN_TEST_DENOISE_LAYER_H
#define TMVA_TEST_DNN_TEST_DENOISE_LAYER_H

#include "../Utility.h"

#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DAE/DenoiseLayer.h"

#include "TMVA/DNN/Functions.h"
#include <iostream>

using namespace TMVA::DNN;
using namespace TMVA::DNN::DAE;

template <typename Architecture> auto testLayer(size_t batchSize, size_t visibleUnits, size_t hiddenUnits)
-> void
{
  //using Scalar_t = typename Architecture::Scalar_t;
  using Matrix_t = typename Architecture::Matrix_t;

  using TDAELayer = TDAELayer<Architecture>;

  TDAELayer dae(batchSize, visibleUnits, hiddenUnits,1, EActivationFunction::kSigmoid);
  dae.Initialize(EInitialization::kUniform);

  std::vector<Matrix_t> input, corruptedInput, compressedInput, reconstructedInput;
  for(size_t i=0; i<batchSize; i++)
  {
    input.emplace_back(visibleUnits,1);
    corruptedInput.emplace_back(visibleUnits,1);
    compressedInput.emplace_back(hiddenUnits,1);
    reconstructedInput.emplace_back(visibleUnits, 1);
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




  corruptedInput = dae.Corruption(input,0.3);
  std::cout<<"Corrupted Input Batch: "<<std::endl;
  for(size_t i=0; i<(size_t)corruptedInput.size(); i++)
  {
    for(size_t j=0; j<(size_t)corruptedInput[i].GetNrows(); j++)
    {
      for(size_t k=0; k<(size_t)corruptedInput[i].GetNcols(); k++)
      {
        std::cout<<corruptedInput[i](j,k)<<"\t";
      }
    std::cout<<std::endl;
    }
  }
  std::cout<<std::endl;




  compressedInput = dae.Encoding(input);
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




  reconstructedInput = dae.Reconstruction();
  std::cout<<"Reconstructed Input Batch: "<<std::endl;
  for(size_t i=0; i<(size_t)reconstructedInput.size(); i++)
  {
    for(size_t j=0; j<(size_t)reconstructedInput[i].GetNrows(); j++)
    {
      for(size_t k=0; k<(size_t)reconstructedInput[i].GetNcols(); k++)
      {
        std::cout<<reconstructedInput[i](j,k)<<"\t";
      }
    std::cout<<std::endl;
    }
  }
std::cout<<std::endl;
dae.Print();
  
  
}
#endif
