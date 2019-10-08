// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Concrete instantiation of the generic backpropagation test for //
// CUDA architectures.                                            //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cuda.h"
#include "TMatrix.h"
#include "TestBackpropagation.h"

using namespace TMVA::DNN;

int main()
{
    using Scalar_t = Double_t;
    TCudaTensor<Scalar_t> dummy(1, 1);  // for initializing Cuda properly

    std::cout << "Testing Backpropagation:" << std::endl;
    double error;
    error = testBackpropagationWeightsLinear<TCuda<Scalar_t>>(1.0);
    if (error > 1e-3)
        return 1;
    error = testBackpropagationL1Regularization<TCuda<Scalar_t>>(1e-2);
    if (error > 1e-3)
        return 1;
    error = testBackpropagationL2Regularization<TCuda<Scalar_t>>(1.0);
    if (error > 1e-3)
        return 1;
    error = testBackpropagationBiasesLinear<TCuda<Scalar_t>>(1.0);
    if (error > 1e-3)
        return 1;
    return 0;
}
