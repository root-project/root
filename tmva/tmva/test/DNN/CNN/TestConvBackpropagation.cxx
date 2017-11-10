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
#ifdef DEBUG
// to debug  the test and print matrices
#define DEBUG_TMVA_TCPUMATRIX
#endif
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestConvNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;


// test first in a simple network with linear activations
bool test1()
{

   size_t batchSizeTest = 2;
   size_t imgDepthTest = 2;  
   size_t imgHeightTest = 4;
   size_t imgWidthTest = 4;
   size_t batchDepth = batchSizeTest;
   size_t batchHeight = imgDepthTest;
   size_t batchWidth = imgWidthTest*  imgHeightTest;
   double stepSize = 1.E-5; // for computing derivatives with finate differences

   ETestType type = kLinearNet; 

   return testConvBackwardPass<TCpu<double>>(batchSizeTest, imgDepthTest, imgHeightTest, imgWidthTest, batchDepth,
                                      batchHeight, batchWidth,stepSize,type);
}
// test in a more complex network
bool test2()
{

   size_t batchSizeTest = 4;
   size_t imgDepthTest = 1;
   size_t imgHeightTest = 8;
   size_t imgWidthTest = 8;
   size_t batchDepth = batchSizeTest;
   size_t batchHeight = imgDepthTest;
   size_t batchWidth = imgHeightTest * imgWidthTest;

   // testConvBackwardPass<TReference<double>>(batchSizeTest1, imgDepthTest1, imgHeightTest1, imgWidthTest1, batchDepth,
   //                                          batchHeight, batchWidth);

   double stepSize = 1.E-5; // for computing derivatives with finate differences
   ETestType type = kRndmActNet; 

   return testConvBackwardPass<TCpu<double>>(batchSizeTest, imgDepthTest, imgHeightTest, imgWidthTest, batchDepth,
                                      batchHeight, batchWidth,stepSize,type);
}

int main()
{
   bool ret = true; 
   std::cout << "Testing CNN Backward Pass:" << std::endl;
   std::cout << "Test1, backward pass with linear activation network - compare with finite difference" << std::endl;

   ret &= test1();
   if (!ret) {
      std::cerr << "ERROR - test1 failed " << std::endl;
      return -1;
   }
   std::cout << "Test2, more complex network architecture no dropout" << std::endl;
   ret &= test2(); 
}

