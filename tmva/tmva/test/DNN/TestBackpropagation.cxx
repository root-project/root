// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Concrete instantiation of the generic backpropagation test for //
// the reference architecture.                                    //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"
#include "TestBackpropagation.h"

using namespace TMVA::DNN;

int main()
{
    std::cout << "Testing Backpropagation:" << std::endl;

    double error;

    //
    // Test backpropagation for linear net.
    //

    error = testBackpropagationWeightsLinear<TReference<double>>(1.0);
    if (error > 1e-3)
        return 1;

    error = testBackpropagationL1Regularization<TReference<double>>(1e-2);
    if (error > 1e-3)
        return 1;

    error = testBackpropagationL2Regularization<TReference<double>>(1.0);
    if (error > 1e-3)
        return 1;

    error = testBackpropagationBiasesLinear<TReference<double>>(1.0);
    if (error > 1e-3)
        return 1;

    return 0;
}
