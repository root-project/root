// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Conv Net Backpropagation                                          *
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
// Testing the Conv Net Backward Pass                             //
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

   testConvBackwardPass<TReference<double>>(batchSizeTest1, imgDepthTest1, imgHeightTest1, imgWidthTest1, batchDepth,
                                            batchHeight, batchWidth);
}

int main()
{
   std::cout << "Testing CNN Backward Pass:" << std::endl;

   std::cout << "Test1, no dropout" << std::endl;
   test1();
}
