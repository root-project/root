// @(#)root/tmva $Id$
// Author: Akshay Vashistha (ajatgd)

/*************************************************************************
 * Copyright (C) 2017, ajatgd                                            *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef TMVA_TEST_DNN_TEST_DAE_TEST_UTILSAE_H_
#define TMVA_TEST_DNN_TEST_DAE_TEST_UTILSAE_H_

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
auto testAddBiases(typename Architecture::Matrix_t & A,
                   typename Architecture::Matrix_t & Biases)
-> void
{
  using Matrix_t = typename Architecture::Matrix_t;
  size_t m,n;
  m=A.GetNrows();
  n=A.GetNcols();

  Architecture::AddBiases(A,Biases);

/*  std::cout<<"Resultant Matrix"<<std::endl;
  for(size_t i=0; i<m; i++)
  {
    for(size_t j=0; j<n; j++)
    {
      std::cout<<A(i,j)<<"\t";
    }
    std::cout<<std::endl;
  }*/
}

//______________________________________________________________________________
template<typename Architecture>
auto testSoftmaxAE(typename Architecture::Matrix_t &A)
-> void
{
  using Matrix_t = typename Architecture::Matrix_t;
  size_t m,n;
  m=A.GetNrows();
  n=A.GetNcols();

  Architecture::SoftmaxAE(A);
  std::cout<<"Resultant Matrix"<<std::endl;

  for(size_t i=0; i<m; i++)
  {
    for(size_t j=0; j<n; j++)
    {
      std::cout<<A(i,j)<<"\t";
    }
    std::cout<<std::endl;
  }
}
#endif
