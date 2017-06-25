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
#include "TMVA/DNN/DAE/denoiseAE.h"

using namespace TMVA::DNN;
using namespavce TMVA::DNN:DAE;




//_____________________________________________________________________________
template<typename Architecture>
auto testCorruptInput(typename Architecture::Matrix_t & input,
                     typename Architecture::Matrix_t & corruptedInput,
                     double corruptionLevel)

-> Matrix_t
{
  size_t m,n;
  m=corruptedInput.GetNrows();
  n=corruptedInput.GetNcols();
  using Matrix_t = typename Architecture::Matrix_t;

  Architecture::corruptedInput(input,corruptedInput,fWeights);

  for (size_t i=0; i<m; i++)
  {
    for (size_t j=0; j<n; j++)
    {
      std::cout<<corruptedInput(i,j)<<std::endl;
    }
    std::cout<<std::endl;
  }

  return corruptedInput;

}


//____________________________________________________________________________


template<typename Architecture>
auto testEncodeInput(typename Architecture::Matrix_t & input,
                     typename Architecture::Matrix_t & compressedInput,
                     typename Architecture::Matrix_t &fWeights)

-> Matrix_t

{
   size_t m,n;
   m=compressedInput.GetNrows();
   n=compressedInput.GetNcols();
   using Matrix_t = typename Architecture::Matrix_t;
   Architecture::EncodeInput(input,compressedInput,fWeights);

   for (size_t i=0; i<m; i++)
   {
     for (size_t j=0; j<n; j++)
     {
       std::cout<<compressedInput(i,j)<<std::endl;
     }
     std::cout<<std::endl;
   }

   return compressedInput;
}

//______________________________________________________________________________

template<typename Architecture>
auto testReconstructInput(typename Architecture::Matrix_t &compressedInput,
                          typename Architecture::Matrix_t &reconstructedInput,
                          typename Architecture::Matrix_t &fWeights)
-> Matrix_t
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
      std::cout<<reconstructedInput(i,j)<<std::endl;
    }
    std::cout<<std::endl;
  }

  return reconstructedInput;


}

//______________________________________________________________________________
