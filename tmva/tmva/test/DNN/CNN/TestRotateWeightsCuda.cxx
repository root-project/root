// @(#)root/tmva/tmva/cnn:$Id$
// Author: Manos Stergiadis

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing RotateWeights method on a CPU architecture                        *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Manos Stergiadis    <em.stergiadis@gmail.com>   - CERN, Switzerland       *
 *                                                                                *
 * Copyright (c) 2005-2015:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include <iostream>
#include <cmath>

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TestRotateWeights.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;

int main()
{
    using Scalar_t = Double_t;

    std::cout << "Testing Rotate Weights function on a GPU architecture:" << std::endl;

    std::cout << "Test 1: " << std::endl;
    if (!test1<TCuda<Scalar_t>>()) {
        std::cerr << "ERROR - Rotate Weights failed " << std::endl;
        return -1;
    }

    std::cout << "Test 2: " << std::endl;
    if (!test2<TCuda<Scalar_t>>()) {
        std::cerr << "ERROR - Rotate Weights failed " << std::endl;
        return -1;
    }

    std::cout << "All tests passed!" << std::endl;
    return 0;
}
