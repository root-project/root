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
    Double_t error = testMultiplication<TCpu<Double_t, false>>(10);
    std::cout << "Testing matrix multiplication: Max. rel. error = " << error
              << std::endl;

   error = testSumColumns<TCpu<Double_t, false>>(10);
    std::cout << "Testing column sum:            Max. rel. error = " << error
              << std::endl;
}
