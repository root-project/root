// @(#)root/tmva $Id$
// Author: Vladimir Ilievski, 20/06/17

/*************************************************************************
 * Copyright (C) 2017, Vladimir Ilievski
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Testing the Conv Net Backward Pass                             //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"

#include "TestConvNet.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;

void test1() {
    
   size_t batchSizeTest1 = 50;
   size_t imgDepthTest1 = 3;
   size_t imgHeightTest1 = 32;
   size_t imgWidthTest1 = 32;
    
   testConvBackwardPass<TReference<double>>(batchSizeTest1, imgDepthTest1,
                                            imgHeightTest1, imgWidthTest1);

}


int main(){
   std::cout << "Testing CNN Backward Pass:" << std::endl;
    
   std::cout << "Test1, no dropout" << std::endl;
   test1();
}
