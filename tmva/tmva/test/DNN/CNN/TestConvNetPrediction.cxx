// @(#)root/tmva $Id$
// Author: Vladimir Ilievski, 16/06/17

/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/


////////////////////////////////////////////////////////////////////
// Testing the Conv Net Prediction                                //
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
   EOutputFunction f = EOutputFunction::kIdentity;
    
    
   testConvPrediction<TReference<double>>(batchSizeTest1, imgDepthTest1,
                                          imgHeightTest1, imgWidthTest1, f);
}

void test2(){
    
   size_t batchSizeTest2 = 50;
   size_t imgDepthTest2 = 3;
   size_t imgHeightTest2 = 32;
   size_t imgWidthTest2 = 32;
   EOutputFunction f = EOutputFunction::kSigmoid;
    
   testConvPrediction<TReference<double>>(batchSizeTest2, imgDepthTest2,
                                          imgHeightTest2, imgWidthTest2, f);
}

void test3(){
    
   size_t batchSizeTest3 = 50;
   size_t imgDepthTest3 = 3;
   size_t imgHeightTest3 = 32;
   size_t imgWidthTest3 = 32;
   ERegularization fR = ERegularization::kL2;
   EOutputFunction f = EOutputFunction::kSoftmax;
    
   testConvPrediction<TReference<double>>(batchSizeTest3, imgDepthTest3,
                                          imgHeightTest3, imgWidthTest3, f);
}

int main(){
   std::cout << "Testing CNN Prediction:" << std::endl;
    
   std::cout << "Test1, identity output function" << std::endl;
   test1();
    
   std::cout << "Test2, sigmoid output function" << std::endl;
   test2();
    
   std::cout << "Test3, softmax output function" << std::endl;
   test3();
}
