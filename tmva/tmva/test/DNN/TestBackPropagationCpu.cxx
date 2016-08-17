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
// multi-threaded CPU architectures.                              //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TMatrix.h"
#include "TestBackpropagation.h"

using namespace TMVA::DNN;

int main()
{
    std::cout << "Testing Backpropagation:" << std::endl;

    double error;

    error = testBackpropagationWeightsLinear<TCpu<Double_t>>(1.0);
    if (error > 1e-7)
        return 1;

    error = testBackpropagationL1Regularization<TCpu<Double_t>>(1e-2);
    if (error > 1e-7)
        return 1;

    error = testBackpropagationL2Regularization<TCpu<Double_t>>(1.0);
    if (error > 1e-7)
        return 1;

    error = testBackpropagationBiasesLinear<TCpu<Double_t>>(1.0);
    if (error > 1e-7)
        return 1;

    return 0;
}
