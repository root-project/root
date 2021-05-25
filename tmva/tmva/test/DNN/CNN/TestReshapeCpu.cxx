// @(#)root/tmva $Id$
// Author: Manos Stergiadis

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Flatten function for Reference backend                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Manos Stergiadis    <em.stergiadis@gmail.com>  - CERN, Switzerland        *
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

////////////////////////////////////////////////////////////////////
// Testing Flatten/Deflatten on the CPU architecture              //
////////////////////////////////////////////////////////////////////

#include <iostream>

#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestReshape.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;

int main()
{
    using Scalar_t = Double_t;
    std::cout << "Testing Flatten/Deflatten on the CPU architecture:" << std::endl;

    std::cout << "Test Reshape: " << std::endl;
    if (!testReshape<TCpu<Scalar_t>>()) {
        std::cerr << "ERROR - testReshape failed " << std::endl;
        return 1;
    }

    std::cout << "Test Flatten: " << std::endl;
    if (!testFlatten<TCpu<Scalar_t>>()) {
        std::cerr << "ERROR - testFlatten failed " << std::endl;
        return 1;
    }

    std::cout << "Test Deflatten: " << std::endl;
    if (!testDeflatten<TCpu<Scalar_t>>()) {
        std::cerr << "ERROR - testDeflatten failed " << std::endl;
        return 1;
    }
    return 0;
}
