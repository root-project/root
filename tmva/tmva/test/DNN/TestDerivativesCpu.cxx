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
// Concrete instantiation of the generic derivative test for the //
//  multi-threaded CPU implementation.                           //
///////////////////////////////////////////////////////////////////

#include <iostream>
#include "RConfigure.h"
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestDerivatives.h"

using namespace TMVA::DNN;

int main()
{
    using Scalar_t = Double_t;

    double error;

    //
    // Activation Functions
    //

    std::cout << "Activation Functions:" << std::endl;
#ifdef R__HAS_VDT
    bool useFastTanh = true;
#else
    bool useFastTanh = false;
#endif
    error = testActivationFunctionDerivatives<TCpu<Scalar_t>>(useFastTanh);
    std::cout << "Total    : ";
    std::cout << "Maximum Relative Error = " << error;
    std::cout << std::endl << std::endl;
    if (error > 1e-3)
        return 1;

    //
    // Loss Functions
    //

    std::cout << "Loss Functions:" << std::endl;
    error = testLossFunctionGradients<TCpu<Scalar_t>>();
    std::cout << "Total    : ";
    std::cout << "Maximum Relative Error = " << error;
    std::cout << std::endl << std::endl;
    if (error > 1e-3)
        return 1;

    //
    // Regularization Functions
    //

    std::cout << "Regularization:" << std::endl;
    error = testRegularizationGradients<TCpu<Scalar_t>>();
    std::cout << "Total    : ";
    std::cout << "Maximum Relative Error = " << error;
    std::cout << std::endl << std::endl;
    if (error > 1e-3)
        return 1;

    return 0;
}
