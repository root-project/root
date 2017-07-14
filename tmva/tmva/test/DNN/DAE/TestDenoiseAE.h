// @(#)root/tmva $Id$
// Author: Akshay Vashistha (ajatgd)

/*************************************************************************
 * Copyright (C) 2017, ajatgd                                            *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef TMVA_TEST_DNN_TEST_DAE_TEST_DENOISEAE_H_
#define TMVA_TEST_DNN_TEST_DAE_TEST_DENOISEAE_H_

////////////////////////////////////////////////////////////////////
// Generic tests of the DAE functionalities                       //
////////////////////////////////////////////////////////////////////


#include <iostream>
#include "../Utility.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DAE/DenoiseAE.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::DAE;




//_____________________________________________________________________________
template<typename Architecture>
auto testCorruptInput(typename Architecture::Matrix_t & input,
                     typename Architecture::Matrix_t & corruptedInput,
                     double corruptionLevel)
-> void
{
  using Matrix_t = typename Architecture::Matrix_t;

  size_t m,n;
  m=input.GetNrows();
  n=input.GetNcols();

  Architecture::CorruptInput(input,corruptedInput,corruptionLevel);

  for (size_t i=0; i<m; i++)
  {
    for (size_t j = 0; j<n; j++)
    {
      std::cout<<corruptedInput(i,j)<<"\t";
    }
    std::cout<<std::endl;
  }

}


//____________________________________________________________________________


template<typename Architecture>
auto testEncodeInput(typename Architecture::Matrix_t & input,
                     typename Architecture::Matrix_t & compressedInput,
                     typename Architecture::Matrix_t & Weights)
-> void
{
   using Matrix_t = typename Architecture::Matrix_t;

   size_t m,n;
   m=compressedInput.GetNrows();
   n=compressedInput.GetNcols();

   Architecture::EncodeInput(input,compressedInput,Weights);

   for (size_t i=0; i<m; i++)
   {
     for (size_t j=0; j<n; j++)
     {
       std::cout<<compressedInput(i,j)<<"\t";
     }
     std::cout<<std::endl;
   }

}

//______________________________________________________________________________

template<typename Architecture>
auto testReconstructInput(typename Architecture::Matrix_t &compressedInput,
                          typename Architecture::Matrix_t &reconstructedInput,
                          typename Architecture::Matrix_t &fWeights)
-> void
{
  size_t m,n;
  m=reconstructedInput.GetNrows();
  n=reconstructedInput.GetNcols();
  using Matrix_t = typename Architecture::Matrix_t;
  Architecture::ReconstructInput(compressedInput,reconstructedInput,fWeights);

  for (size_t i=0; i<m; i++)
  {
    for (size_t j=0; j<n; j++)
    {
      std::cout<<reconstructedInput(i,j)<<"\t";
    }
    std::cout<<std::endl;
  }
}
//______________________________________________________________________________
template<typename Architecture>
auto testUpdateParameter(typename Architecture::Matrix_t &x,
                         typename Architecture::Matrix_t &tildeX,
                         typename Architecture::Matrix_t &y,
                         typename Architecture::Matrix_t &z,
                         typename Architecture::Matrix_t &fVBiases,
                         typename Architecture::Matrix_t &fHBiases,
                         typename Architecture::Matrix_t &fWeights,
                         typename Architecture::Matrix_t &VBiasError,
                         typename Architecture::Matrix_t &HBiasError,
                         double learningRate,
                         size_t fBatchSize)
-> void
{
  using Matrix_t = typename Architecture::Matrix_t;
  size_t m,n;
  m=fWeights.GetNrows();
  n=fWeights.GetNcols();
  Architecture::UpdateParams(x,tildeX,y,z,fVBiases,fHBiases,fWeights,VBiasError,HBiasError,learningRate,fBatchSize);
  for (size_t i=0; i<m; i++)
  {
    for (size_t j=0; j<n; j++)
    {
      std::cout<<fWeights(i,j)<<"\t";
    }
    std::cout<<std::endl;
  }
}

//______________________________________________________________________________
#endif
