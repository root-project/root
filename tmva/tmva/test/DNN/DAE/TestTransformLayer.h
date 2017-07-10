// @(#)root/tmva $Id$
// Author: Akshay Vashistha (ajatgd)

/*************************************************************************
 * Copyright (C) 2017, ajatgd                                            *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


#ifndef TMVA_TEST_DNN_TEST_DAE_TEST_TRANSFORMLAYER_H_
#define TMVA_TEST_DNN_TEST_DAE_TEST_TRANSFORMLAYER_H_

////////////////////////////////////////////////////////////////////
// Generic tests of the DAE functionalities                       //
////////////////////////////////////////////////////////////////////


#include <iostream>
#include "../Utility.h"
#include "TMVA/DNN/Functions.h"
#include "TMVA/DNN/DAE/StackedNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::DAE;




//_____________________________________________________________________________
template<typename Architecture>
auto testTransform()
-> typename Architecture::Matrix_t
{
  using Matrix_t = typename Architecture::Matrix_t;

  TMatrixT<Double_t> input(5,1);
  //TMatrixT<Double_t> transformed(2, 1);
  TMatrixT<Double_t> fWeights(6, 5);
//  TMatrixT<Double_t> fBiases(6,1);
  randomMatrix(input);
  randomMatrix(fWeights);

  Matrix_t Input(input);


  Matrix_t transformed(6,1);

  Matrix_t Weights(fWeights);


  Matrix_t fBiases(6,1);

  for(size_t i=0; i<(size_t)fBiases.GetNrows();i++)
  {
    for (size_t j = 0; j < (size_t)fBiases.GetNcols(); j++)
    {
      fBiases(i,j)=0;
    }
  }

  size_t m,n;
  m=transformed.GetNrows();
  n=transformed.GetNcols();

  Architecture::Transform(input,transformed,fWeights,fBiases);

  for(size_t i=0; i<m;i++)
  {
    for(size_t j=0; j<n; j++)
    {
      std::cout<<transformed(i,j)<<"\t";
    }
    std::cout<<std::endl;
  }
  return transformed;
}
#endif
