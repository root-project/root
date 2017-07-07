// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////
// Test for the loss function reference implementation using the //
// generic test defined in TestLossFunctions.h.                  //
///////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Reference.h"
#include "TestLossFunctions.h"

using namespace TMVA::DNN;

int main()
{
    std::cout << "Testing Loss Functions:" << std::endl << std::endl;

    double error;

    //
    // Mean Squared Error.
    //

    error = testMeanSquaredError<TReference<double>>(10);
    std::cout << "Testing mean squared error loss:        ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testMeanSquaredErrorGradients<TReference<double>>(10);
    std::cout << "Testing mean squared error gradient:    ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    //
    // Cross Entropy.
    //

    error = testCrossEntropy<TReference<double>>(10);
    std::cout << "Testing cross entropy loss:             ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testCrossEntropyGradients<TReference<double>>(10);
    std::cout << "Testing cross entropy gradient:         ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    //
    // Softmax Cross Entropy.
    //

    error = testSoftmaxCrossEntropy<TReference<double>>(10);
    std::cout << "Testing softmax cross entropy loss:     ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-3) return 1;

    error = testSoftmaxCrossEntropyGradients<TReference<double>>(10);
    std::cout << "Testing softmax cross entropy gradient: ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-3) return 1;
}
