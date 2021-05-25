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
#include "RConfigure.h"

using namespace TMVA::DNN;

int main()
{
    using Scalar_t = Double_t;
    std::cout << "Testing Activation Functions:" << std::endl;

    Scalar_t error;

    // Identity.

    error = testIdentity<TReference<Scalar_t>>(10);
    std::cout << "Testing identity activation:               ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testIdentityDerivative<TReference<Scalar_t>>(10);
    std::cout << "Testing identity activation derivative:    ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    // ReLU.

    error = testRelu<TReference<Scalar_t>>(10);
    std::cout << "Testing ReLU activation:                   ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testReluDerivative<TReference<Scalar_t>>(10);
    std::cout << "Testing ReLU activation derivative:        ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    // Sigmoid.

    error = testSigmoid<TReference<Scalar_t>>(10);
    std::cout << "Testing Sigmoid activation:                ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testSigmoidDerivative<TReference<Scalar_t>>(10);
    std::cout << "Testing Sigmoid activation derivative:     ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    // TanH.

    error = testTanh<TReference<Scalar_t>>(10);
    std::cout << "Testing TanH activation:                   ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
#ifdef R__HAS_VDT   // error is larger when using fast tanh from vdt
    std::cout << "error is " << error << std::endl;
    if (error > 1e-6) return 1;
#else
    std::cout << "no vdt: error is " << error << std::endl;
    if (error > 1e-10) return 1;
#endif

    error = testTanhDerivative<TReference<Scalar_t>>(10);
    std::cout << "Testing TanH activation derivative:        ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
#ifdef R__HAS_VDT   // error is larger when using fast tanh from vdt
    if (error > 1e-4) return 1; 
#else
    if (error > 1e-10) return 1; 
#endif

    // Symmetric ReLU.

    error = testSymmetricReluDerivative<TReference<Scalar_t>>(10);
    std::cout << "Testing Symm. ReLU activation:             ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testSymmetricReluDerivative<TReference<Scalar_t>>(10);
    std::cout << "Testing Symm. ReLU activation derivative:  ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    // Soft Sign.

    error = testSoftSign<TReference<Scalar_t>>(10);
    std::cout << "Testing Soft Sign activation:              ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testSoftSignDerivative<TReference<Scalar_t>>(10);
    std::cout << "Testing Soft Sign activation derivative:   ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    // Gauss.

    error = testGauss<TReference<Scalar_t>>(10);
    std::cout << "Testing Gauss activation:                  ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testGaussDerivative<TReference<Scalar_t>>(10);
    std::cout << "Testing Gauss activation derivative:       ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    return 0;
}
