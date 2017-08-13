// @(#)root/tmva/tmva/cnn:$Id$
// Author: Vladimir Ilievski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Deep Learning Minimizer for the Reference backend                 *
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

#include "TestMinimization.h"
#include "TMVA/DNN/Architectures/Reference.h"

#include <iostream>

using namespace TMVA::DNN;

int main()
{
   using Scalar_t = Real_t;

   Scalar_t momentum = 0.0;
   std::cout << "Minimizer, no momentum" << std::endl;
   testMinimization<TReference<Scalar_t>>(momentum, false);

   momentum = 0.1;
   std::cout << "Minimizer, with momentum" << std::endl;
   testMinimization<TReference<Scalar_t>>(momentum, false);

   std::cout << "Minimizer, with Nestorov momentum" << std::endl;
   testMinimization<TReference<Scalar_t>>(momentum, true);
}