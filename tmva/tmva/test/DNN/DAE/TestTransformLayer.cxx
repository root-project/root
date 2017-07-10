// @(#)root/tmva $Id$
// Author: Akshay Vashistha (ajatgd)

/*************************************************************************
 * Copyright (C) 2017, ajatgd
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


////////////////////////////////////////////////////////////////////
// Testing the Encode function                                    //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cmath>

#include "TMVA/DNN/Architectures/Reference.h"
#include "TestTransformLayer.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::DAE;

//using Matrix_t = typename TReference<double>::Matrix_t;

/*void test1()
{
  Matrix_t input(5,1);
  randomMatrix(input);

  Matrix_t transformed(6,1);

  Matrix_t fWeights(6,5);
  randomMatrix(fWeights);

  Matrix_t fBiases(6,1);

  for(size_t i=0; i<(size_t)fBiases.GetNrows();i++)
  {
    for (size_t j = 0; j < (size_t)fBiases.GetNcols(); j++)
    {
      fBiases(i,j)=0;
    }
  }

  std::cout<<"Transformed Matrix"  << std::endl;
  testTransform<TReference<double>>(input,transformed,fWeights,fBiases);

}

void test2()
{
  Matrix_t input(5,1);
  randomMatrix(input);

  Matrix_t transformed(2,1);

  Matrix_t fWeights(2,5);
  randomMatrix(fWeights);

  Matrix_t fBiases(2,1);

  for(size_t i=0; i<(size_t)fBiases.GetNrows();i++)
  {
    for (size_t j = 0; j < (size_t)fBiases.GetNcols(); j++)
    {
      fBiases(i,j)=0;
    }
  }

  std::cout<<"Transformed Matrix"  << std::endl;
  testTransform<TReference<double>>(input,transformed,fWeights,fBiases);

}*/
using Matrix_t = typename TReference<double>::Matrix_t;

int main()
{
  //test1();
  //test2();

  std::cout<<"Testing transform Layer"<<std::endl<<std::endl;
  Matrix_t A(6,1);
  A=(testTransform<TReference<double>>());
  //testReconstructInput<TReference<double>>();
  std::cout<< "Testing transform:    "<<std::endl;

  size_t m,n;
  m=A.GetNrows();
  n=A.GetNcols();
  for(size_t i; i<m;i++)
  {
    for(size_t j;j<n;j++)
    {
      std::cout<<A(i,j)<<"\t";
    }
    std::cout<<std::endl;
  }
}
