// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////
//  Concrete instantiation of the generic activation function test  //
//  for the reference architecture.                                 //
//////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TestActivationFunctions.h"

using namespace TMVA::DNN;

int main()
{
    std::cout << "Testing Activation Functions:" << std::endl;

    double error;

    // Identity.

    error = testIdentity<TReference<double>>(10);
    std::cout << "Testing identity activation:               ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testIdentityDerivative<TReference<double>>(10);
    std::cout << "Testing identity activation derivative:    ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    // ReLU.

    error = testRelu<TReference<double>>(10);
    std::cout << "Testing ReLU activation:                   ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testReluDerivative<TReference<double>>(10);
    std::cout << "Testing ReLU activation derivative:        ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    // Sigmoid.

    error = testSigmoid<TReference<double>>(10);
    std::cout << "Testing Sigmoid activation:                ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testSigmoidDerivative<TReference<double>>(10);
    std::cout << "Testing Sigmoid activation derivative:     ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    // TanH.

    error = testTanh<TReference<double>>(10);
    std::cout << "Testing TanH activation:                   ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testTanhDerivative<TReference<double>>(10);
    std::cout << "Testing TanH activation derivative:        ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    // Symmetric ReLU.

    error = testSymmetricRelu<TReference<double>>(10);
    std::cout << "Testing Symm. ReLU activation:             ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testSymmetricRelu<TReference<double>>(10);
    std::cout << "Testing Symm. ReLU activation derivative:  ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    // Soft Sign.

    error = testSoftSign<TReference<double>>(10);
    std::cout << "Testing Soft Sign activation:              ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testSoftSign<TReference<double>>(10);
    std::cout << "Testing Soft Sign activation derivative:   ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    // Gauss.

    error = testGauss<TReference<double>>(10);
    std::cout << "Testing Gauss activation:                  ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testGauss<TReference<double>>(10);
    std::cout << "Testing Gauss activation derivative:       ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    return 0;
}
