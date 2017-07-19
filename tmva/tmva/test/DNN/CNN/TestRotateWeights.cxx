// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing RotateWeights method                                              *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Vladimir Ilievski      <ilievski.vladimir@live.com>  - CERN, Switzerland  *
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

////////////////////////////////////////////////////////////////////
// Testing the Rotate Weights function                            //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include <cmath>

#include "TMVA/DNN/Architectures/Reference.h"
#include "TestConvNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;
using Matrix_t = typename TReference<double>::Matrix_t;

/*************************************************************************
 * Test 1:
 *  filter depth = 3, filter height = 2, filter width = 2, num. filters = 4
 *************************************************************************/
void test1()
{
   double weightsTest1[][12] = {{252, 116, 155, 246, 170, 149, 227, 113, 166, 227, 119, 57},
                                {92, 103, 151, 37, 110, 46, 70, 8, 88, 182, 43, 236},
                                {153, 246, 216, 102, 179, 248, 187, 227, 66, 102, 180, 169},
                                {5, 215, 115, 103, 35, 138, 193, 28, 213, 93, 117, 208}};

   double answerTest1[][16] = {{246, 155, 116, 252, 37, 151, 103, 92, 102, 216, 246, 153, 103, 115, 215, 5},
                               {113, 227, 149, 170, 8, 70, 46, 110, 227, 187, 248, 179, 28, 193, 138, 35},
                               {57, 119, 227, 166, 236, 43, 182, 88, 169, 180, 102, 66, 208, 117, 93, 213}};

   size_t filterDepthTest1 = 3;
   size_t filterHeightTest1 = 2;
   size_t filterWidthTest1 = 2;
   size_t numFiltersTest1 = 4;

   Matrix_t A(numFiltersTest1, filterDepthTest1 * filterHeightTest1 * filterWidthTest1);

   for (size_t i = 0; i < (size_t)A.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t)A.GetNcols(); j++) {
         A(i, j) = weightsTest1[i][j];
      }
   }

   Matrix_t B(filterDepthTest1, numFiltersTest1 * filterHeightTest1 * filterWidthTest1);

   for (size_t i = 0; i < (size_t)B.GetNrows(); i++) {
      for (size_t j = 0; j < (size_t)B.GetNcols(); j++) {
         B(i, j) = answerTest1[i][j];
      }
   }

   bool status = testRotateWeights<TReference<double>>(A, B, filterDepthTest1, filterHeightTest1, filterWidthTest1,
                                                       numFiltersTest1);

   if (status)
      std::cout << "Test passed!" << std::endl;
   else
      std::cout << "Test not passed!" << std::endl;
}

int main()
{
   std::cout << "Testing Rotate Weights function:" << std::endl;

   std::cout << "Test 1: " << std::endl;
   test1();
}
