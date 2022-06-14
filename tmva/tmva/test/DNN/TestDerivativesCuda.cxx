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
//  reference implementation.                                    //
///////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cuda.h"
#include "TestDerivatives.h"

using namespace TMVA::DNN;

int main()
{
    using Scalar_t = Double_t;
    Double_t error;
    TCudaTensor<Scalar_t> dummy(1, 1);
    //
    // Activation Functions
    //

    std::cout << "Activation Functions:" << std::endl;
    error = testActivationFunctionDerivatives<TCuda<Scalar_t>>();
    std::cout << "Total    : ";
    std::cout << "Maximum Relative Error = " << error;
    std::cout << std::endl << std::endl;
    if (error > 1e-2)
        return 1;

    //
    // Loss Functions
    //

    std::cout << "Loss Functions:" << std::endl;
    error = testLossFunctionGradients<TCuda<Scalar_t>>();
    std::cout << "Total    : ";
    std::cout << "Maximum Relative Error = " << error;
    std::cout << std::endl << std::endl;
    if (error > 1e-3)
        return 1;

    //
    // Regularization Functions
    //

    std::cout << "Regularization:" << std::endl;
    error = testRegularizationGradients<TCuda<Scalar_t>>();
    std::cout << "Total    : ";
    std::cout << "Maximum Relative Error = " << error;
    std::cout << std::endl << std::endl;
    if (error > 1e-3)
        return 1;

    return 0;
}
