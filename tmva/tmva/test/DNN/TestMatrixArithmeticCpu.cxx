// @(#)root/tmva/tmva/dnn:$Id$ // Author: Simon Pfreundschuh 20/07/16

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

///////////////////////////////////////////////////////////////////
// Test arithmetic on CpuMatrix class using the generic tests in //
// TestArithmetic.h                                              //
///////////////////////////////////////////////////////////////////

#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestMatrixArithmetic.h"

using namespace TMVA::DNN;

int main()
{
    std::cout << "Testing CPU matrix arithmetic (double):" << std::endl;

    Double_t error = testMultiplication<TCpu<Double_t>>(10);
    std::cout << "Multiplication: " << "Max. rel. error: " << error << std::endl;
    if (error > 1e-3)
        return 1;

    error = testSumColumns<TCpu<Double_t>>(1);
    std::cout << "Column Sum:     " << "Max. rel. error: " << error << std::endl;
    if (error > 1e-3)
        return 1;

    std::cout << "Testing CPU matrix arithmetic (float):" << std::endl;

    error = testMultiplication<TCpu<Real_t>>(10);
    std::cout << "Multiplication: " << "Max. rel. error: " << error << std::endl;
    if (error > 1e-1)
        return 1;

    error = testSumColumns<TCpu<Real_t>>(1);
    std::cout << "Column Sum:     " << "Max. rel. error: " << error << std::endl;
    if (error > 1e-1)
        return 1;
}
