// @(#)root/tmva/tmva/test/dnn $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////
// Concrete instantiation of the generic activation function test   //
// for the TCuda implementation.                                    //
//////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cuda.h"
#include "Utility.h"
#include "TestActivationFunctions.h"

using namespace TMVA::DNN;

int main()
{
    using Scalar_t = Double_t;

    std::cout << "Testing Activation Functions:" << std::endl;

    double error;

    // Identity.

    error = testIdentity<TCuda<Scalar_t>>(10);
    std::cout << "Testing identity activation:            ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-5)
        return 1;

#if 0 // fix to not use reference arch
    error = testIdentityDerivative<TCuda<Scalar_t>>(10);
    std::cout << "Testing identity activation derivative: ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-5)
        return 1;
#endif

    // ReLU.

    error = testRelu<TCuda<Scalar_t>>(10);
    std::cout << "Testing ReLU activation:                ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-5)
        return 1;

    error = testReluDerivative<TCuda<Scalar_t>>(10);
    std::cout << "Testing ReLU activation derivative:     ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-5)
        return 1;

    // Sigmoid.

    error = testSigmoid<TCuda<Scalar_t>>(10);
    std::cout << "Testing Sigmoid activation:             ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-5)
        return 1;

    error = testSigmoidDerivative<TCuda<Scalar_t>>(10);
    std::cout << "Testing Sigmoid activation derivative:  ";
    std::cout << "maximum relative error = " << error << std::endl;
    if (error > 1e-5)
        return 1;
    return 0;
}
