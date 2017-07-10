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
#include "TMVA/DNN/Functions.h"
#include "TestUtilsAE.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::DAE;

using Matrix_t = typename TReference<double>::Matrix_t;

void test1()
{
  Matrix_t A(5,1);
  randomMatrix(A);

  std::cout<<"Given Matrix"<<std::endl;
  for(size_t i=0; i < (size_t)A.GetNrows(); i++)
  {
    for(size_t j=0; j < (size_t) A.GetNcols();j++)
    {
      std::cout<<A(i,j)<<"\t";
    }
    std::cout<<std::endl;
  }
  std::cout<<std::endl;

  Matrix_t Biases(5,1);
  randomMatrix(Biases);

  std::cout<<"Bias Matrix"<<std::endl;
  for(size_t i=0; i < (size_t)Biases.GetNrows(); i++)
  {
    for(size_t j=0; j < (size_t) Biases.GetNcols();j++)
    {
      std::cout<<Biases(i,j)<<"\t";
    }
    std::cout<<std::endl;
  }
  std::cout<<std::endl;

  std::cout<<"Adding Bias matrix to given Matrix"<<std::endl;
  testAddBiases<TReference<double>>(A,Biases);
  std::cout<<std::endl;
  std::cout<<"Given Matrix after Softmax Operation"<<std::endl;
  testSoftmaxAE<TReference<double>>(A);
  std::cout<<std::endl;
}
void test2()
{
  Matrix_t A(5,4);
  randomMatrix(A);

  std::cout<<"Given Matrix"<<std::endl;
  for(size_t i=0; i < (size_t)A.GetNrows(); i++)
  {
    for(size_t j=0; j < (size_t) A.GetNcols();j++)
    {
      std::cout<<A(i,j)<<"\t";
    }
    std::cout<<std::endl;
  }
  std::cout<<std::endl;

  Matrix_t Biases(5,1);
  randomMatrix(Biases);

  std::cout<<"Bias Matrix"<<std::endl;
  for(size_t i=0; i < (size_t)Biases.GetNrows(); i++)
  {
    for(size_t j=0; j < (size_t) Biases.GetNcols();j++)
    {
      std::cout<<Biases(i,j)<<"\t";
    }
    std::cout<<std::endl;
  }
  std::cout<<std::endl;

  std::cout<<"Given Matrix after Softmax Operation"<<std::endl;
  testSoftmaxAE<TReference<double>>(A);
  std::cout<<std::endl;

  std::cout<<"Adding Bias matrix to given Matrix"<<std::endl;
  testAddBiases<TReference<double>>(A,Biases);
  std::cout<<std::endl;


}

int main()
{
  test1();
  test2();
}
