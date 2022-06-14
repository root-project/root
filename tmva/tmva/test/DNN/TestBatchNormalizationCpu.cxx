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
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestBatchNormalization.h"

using namespace TMVA::DNN;

int test()
{
    std::cout << "Testing Backpropagation:" << std::endl;

    double error;

    gRandom->SetSeed(111);
    TCpu<double>::SetRandomSeed(gRandom->Integer(TMath::Limits<UInt_t>::Max()));
    //
    // Test backpropagation for linear net.
    //

    error = testBackpropagationWeights<TCpu<double>>(0.0001);
    if (error > 1e-3)
        return 1;

    error = testCNNBackpropagationWeights<TCpu<double>>(0.0001);
    if (error > 1e-3)
       return 1;

    return 0;
}

int main() {
    return test();
}
