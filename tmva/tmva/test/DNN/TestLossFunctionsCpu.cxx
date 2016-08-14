// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////
// Test for the loss function implementatoins for the           //
// multi-threaded CPU version using the generic test defined in //
// TestLossFunctions.h.                                         //
//////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestLossFunctions.h"

using namespace TMVA::DNN;

int main()
{
    std::cout << "Testing Loss Functions:" << std::endl << std::endl;

    double error;

    //
    // Mean Squared Error.
    //

    error = testMeanSquaredError<TCpu<Double_t, false>>(10);
    std::cout << "Testing mean squared error loss:     ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testMeanSquaredErrorGradients<TCpu<Double_t, false>>(10);
    std::cout << "Testing mean squared error gradient: ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    //
    // Cross Entropy.
    //

    error = testCrossEntropy<TCpu<Double_t, false>>(10);
    std::cout << "Testing cross entropy loss:          ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;

    error = testCrossEntropyGradients<TCpu<Double_t, false>>(10);
    std::cout << "Testing mean squared error gradient: ";
    std::cout << "maximum relative error = " << print_error(error) << std::endl;
    if (error > 1e-10)
        return 1;
}
