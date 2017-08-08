// @(#)root/tmva $Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Flatten function for Reference backend                            *
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
// Testing the Flatten Function                                   //
////////////////////////////////////////////////////////////////////

#include <iostream>

#include "TMVA/DNN/Architectures/Reference.h"
#include "TestConvNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;
using Matrix_t = typename TReference<double>::Matrix_t;

/*************************************************************************
 * Test 1:
 * depth = 3, width = 5, height = 5
 *************************************************************************/

void test1()
{

   double imgTest1[][5][5] = {{
                                 {158, 157, 22, 166, 179},
                                 {68, 179, 233, 110, 163},
                                 {168, 216, 76, 8, 102},
                                 {159, 163, 25, 78, 119},
                                 {116, 50, 206, 102, 247},
                              },

                              {{187, 166, 121, 112, 136},
                               {237, 30, 180, 7, 248},
                               {52, 172, 146, 130, 92},
                               {124, 244, 214, 175, 9},
                               {80, 232, 139, 224, 237}},

                              {{53, 147, 103, 53, 110},
                               {112, 222, 19, 156, 232},
                               {81, 19, 188, 224, 220},
                               {255, 190, 76, 219, 95},
                               {245, 4, 217, 22, 22}}};

   double answerTest1[][25] = {{158, 157, 22,  166, 179, 68, 179, 233, 110, 163, 168, 216, 76,
                                8,   102, 159, 163, 25,  78, 119, 116, 50,  206, 102, 247},

                               {187, 166, 121, 112, 136, 237, 30, 180, 7,   248, 52,  172, 146,
                                130, 92,  124, 244, 214, 175, 9,  80,  232, 139, 224, 237},

                               {53,  147, 103, 53,  110, 112, 222, 19,  156, 232, 81, 19, 188,
                                224, 220, 255, 190, 76,  219, 95,  245, 4,   217, 22, 22}};

   size_t sizeTest1 = 3;
   size_t nRowsTest1 = 5;
   size_t nColsTest1 = 5;

   std::vector<Matrix_t> A;
   for (size_t i = 0; i < sizeTest1; i++) {
      Matrix_t temp(nRowsTest1, nColsTest1);
      for (size_t j = 0; j < nRowsTest1; j++) {
         for (size_t k = 0; k < nColsTest1; k++) {
            temp(j, k) = imgTest1[i][j][k];
         }
      }
      A.push_back(temp);
   }

   Matrix_t B(sizeTest1, nRowsTest1 * nColsTest1);
   for (size_t i = 0; i < sizeTest1; i++) {
      for (size_t j = 0; j < nRowsTest1 * nColsTest1; j++) {
         B(i, j) = answerTest1[i][j];
      }
   }

   bool status = testFlatten<TReference<double>>(A, B, sizeTest1, nRowsTest1, nColsTest1);

   if (status)
      std::cout << "Test passed!" << std::endl;
   else
      std::cout << "Test not passed!" << std::endl;
}

int main()
{
   std::cout << "Testing Flatten function:" << std::endl;

   std::cout << "Test 1: " << std::endl;
   test1();
}
