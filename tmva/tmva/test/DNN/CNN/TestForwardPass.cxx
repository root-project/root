// @(#)root/tmva $Id$
// Author: Vladimir Ilievski, 15/06/17

/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Testing the Conv Net Forward Pass                              //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"

#include "TestConvNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;


void test1(){
    
   size_t batchSizeTest1 = 50;
   size_t imgDepthTest1 = 3;
   size_t imgHeightTest1 = 32;
   size_t imgWidthTest1 = 32;
    
   testConvForwardPass<TReference<double>>(batchSizeTest1, imgDepthTest1,
                                           imgHeightTest1, imgWidthTest1);
}

void test2(){
    
   size_t batchSizeTest2 = 50;
   size_t imgDepthTest2 = 3;
   size_t imgHeightTest2 = 32;
   size_t imgWidthTest2 = 32;
   double dropoutProbTest2 = 0.5;
   bool applyDropoutTest2 = true;
    
   testConvForwardPass<TReference<double>>(batchSizeTest2, imgDepthTest2,
                                           imgHeightTest2, imgWidthTest2,
                                           dropoutProbTest2, applyDropoutTest2);
}

int main(){
   std::cout << "Testing CNN Forward Pass:" << std::endl;
    
   std::cout << "Test1, no dropout" << std::endl;
   test1();
    
   std::cout << "Test2, with dropout" << std::endl;
   test2();
}
