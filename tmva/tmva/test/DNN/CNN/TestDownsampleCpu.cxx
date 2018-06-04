// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Downsample method on a CPU architecture                           *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Vladimir Ilievski      <ilievski.vladimir@live.com>  - CERN, Switzerland  *
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
// Testing the Downsample function                                //
////////////////////////////////////////////////////////////////////
#include <iostream>
#include <cmath>

#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestDownsample.h"

int main()
{
    using Scalar_t = Double_t;

    std::cout << "Testing Downsample on the CPU:" << std::endl;

    bool status = true;

    std::cout << "Test 1: " << std::endl;
    status &= test1<TCpu<Scalar_t>>();
    if (!status) {
        std::cerr << "ERROR - test1 failed " << std::endl;
        return -1;
    }

    std::cout << "Test 2: " << std::endl;
    status &= test2<TCpu<Scalar_t>>();
    if (!status) {
        std::cerr << "ERROR - test2 failed " << std::endl;
        return -1;
    }
}