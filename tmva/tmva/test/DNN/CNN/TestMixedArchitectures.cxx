// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Conv Net Forward Pass for the CPU                                 *
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
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TMVA/DNN/Architectures/TCudnn.h"

#include "TestConvNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;


bool test1(bool debug)
{

   size_t batchSizeTest1 = 3;
   size_t imgDepthTest1 = 2;
   size_t imgHeightTest1 = 4;
   size_t imgWidthTest1 = 4;
   size_t batchDepth = batchSizeTest1;
   size_t batchHeight = imgDepthTest1;
   size_t batchWidth = imgHeightTest1 * imgWidthTest1;

   float error = testMixedConvForwardPass<TCudnn<float>, TCpu<float>>(
      batchSizeTest1, imgDepthTest1, imgHeightTest1, imgWidthTest1, batchDepth, batchHeight, batchWidth, debug);

   if (error > 1.E-4) {
      Error("TestMixedArchitecture", " Test of architecture from Cudnn to Cpu failed");
      return false;
   }
   Info("TestMixedArchitecture", "Test passed !!! Same output is obtained from Cudnn -> CPu");
   return true;
}

bool test2(bool debug)
{

   size_t batchSizeTest1 = 3;
   size_t imgDepthTest1 = 2;
   size_t imgHeightTest1 = 4;
   size_t imgWidthTest1 = 4;
   size_t batchDepth = batchSizeTest1;
   size_t batchHeight = imgDepthTest1;
   size_t batchWidth = imgHeightTest1 * imgWidthTest1;

   float error = testMixedConvForwardPass<TCpu<float>, TCudnn<float>>(
      batchSizeTest1, imgDepthTest1, imgHeightTest1, imgWidthTest1, batchDepth, batchHeight, batchWidth, debug);

   if (error > 1.E-4) {
      Error("TestMixedArchitecture", " Test of architecture from Cpu to Cudnn failed");
      return false;
   }
   Info("TestMixedArchitecture", "Test passed !!! Same output is obtained from Cpu -> Cudnn");
   return true;
}

int main()
{

   bool debug = false;
   std::cout << "Testing CNN Forward Pass for GPU (Cudnn)  - CPU:" << std::endl;

   std::cout << "Test1" << std::endl;
   bool ok = test1(debug);

   std::cout << "Test2" << std::endl;
   ok &= test2(debug);

   return (ok) ? 0 : -1;

}
