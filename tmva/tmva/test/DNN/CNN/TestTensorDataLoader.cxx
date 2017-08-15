// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Tensor Data Loader Features                                       *
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

//////////////////////////////////////////////////////////
// Test the reference tensor data loader implementation. //
//////////////////////////////////////////////////////////

#include "TestTensorDataLoader.h"
#include "TMVA/DNN/Architectures/Reference.h"

using namespace TMVA::DNN;

int main()
{

   using Scalar_t = Real_t;

   std::cout << "Testing tensor data loader:" << std::endl;

   //    Scalar_t maximumError = 0.0;
   //    Scalar_t error = testSum<TReference<Scalar_t>>();
   //    std::cout << "Sum: Maximum relative error = " << error << std::endl;
   //    maximumError = std::max(error, maximumError);

   //    error = testIdentity<TReference<Scalar_t>>();
   //    std::cout << "Identity: Maximum relative error = " << error << std::endl;
   //    maximumError = std::max(error, maximumError);

   testDataLoaderDataSet<TReference<Scalar_t>>();
}