// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////
// Test the Neural Network training using the reference   //
// implementation.                                        //
//                                                        //
// Calls the generic testMinimization function defined in //
// TestMinimization.cpp for the reference architecture.   //
////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"
#include "TestMinimization.h"

using namespace TMVA::DNN;

int main()
{
    testMinimization<TReference<double>>();
}
