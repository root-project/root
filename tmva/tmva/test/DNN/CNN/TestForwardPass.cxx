// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Conv Net Forward Pass                                             *
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
// Testing the Conv Net Forward Pass                              //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"

#include "TestConvNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;

void test1()
{

   size_t batchSizeTest1 = 50;
   size_t imgDepthTest1 = 3;
   size_t imgHeightTest1 = 32;
   size_t imgWidthTest1 = 32;
   size_t batchDepth = batchSizeTest1;
   size_t batchHeight = imgDepthTest1;
   size_t batchWidth = imgHeightTest1 * imgWidthTest1;

   testConvForwardPass<TReference<double>>(batchSizeTest1, imgDepthTest1, imgHeightTest1, imgWidthTest1, batchDepth,
                                           batchHeight, batchWidth);
}

void test2()
{

   size_t batchSizeTest2 = 50;
   size_t imgDepthTest2 = 3;
   size_t imgHeightTest2 = 32;
   size_t imgWidthTest2 = 32;
   size_t batchDepth = batchSizeTest2;
   size_t batchHeight = imgDepthTest2;
   size_t batchWidth = imgHeightTest2 * imgWidthTest2;

   testConvForwardPass<TReference<double>>(batchSizeTest2, imgDepthTest2, imgHeightTest2, imgWidthTest2, batchDepth,
                                           batchHeight, batchWidth);
}

int main()
{
   std::cout << "Testing CNN Forward Pass:" << std::endl;

   std::cout << "Test1" << std::endl;
   test1();

   std::cout << "Test2" << std::endl;
   test2();
}
