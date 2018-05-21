/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Im2Col method on the GPU                                          *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Manos Stergiadis      <em.stergiadis@gmail.com>  - CERN, Switzerland      *
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
// Concrete instantiation of the generic Im2Col test for          //
// CUDA architectures.                                            //
////////////////////////////////////////////////////////////////////
#include <iostream>

#include "TMVA/DNN/Architectures/Cuda.h"
#include "TestIm2Col.h"

using namespace TMVA::DNN;
using namespace TMVA::DNN::CNN;


int main()
{
    using Scalar_t = Double_t;
    std::cout << "Testing Im2Col function on the GPU:" << std::endl;

    std::cout << "Test 1: " << std::endl;
    test1<TCuda<Scalar_t>>();

    std::cout << "Test 2: " << std::endl;
    test2<TCuda<Scalar_t>>();

    std::cout << "Test 3: " << std::endl;
    test3<TCuda<Scalar_t>>();
}