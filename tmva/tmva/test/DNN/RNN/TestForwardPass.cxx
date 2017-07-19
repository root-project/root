// @(#)root/tmva $Id$
// Author: Saurav Shekhar 22/06/17

/*************************************************************************
 * Copyright (C) 2017, Saurav Shekhar                                    *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
//Testing RNNLayer forward pass for Reference implementation         //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"
#include "TestForwardPass.h"
//#include "gtest/gtest.h"
//#include "gmock/gmock.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::RNN;

//TEST(RNNTest, ForwardPass)
//{
// EXPECT_EQ(testForwardPass<TReference<double>>(3, 8, 100, 50), 0.0);
//}

int main() {
   std::cout << "Testing RNN Forward pass\n";

   std::cout << testForwardPass<TReference<double>>(1, 8, 100, 50)  << "\n";
   std::cout << testForwardPass<TReference<double>>(5, 9, 128, 64)  << "\n";

   return 0;
}
