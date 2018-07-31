// @(#)root/tmva/tmva/dnn:$Id$
// Author: Ravi Kiran S

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  :                                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Testing Stochastic Batch Gradient Descent Optimizer for Cpu Backend       *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Ravi Kiran S      <sravikiran0606@gmail.com>  - CERN, Switzerland         *
 *                                                                                *
 * Copyright (c) 2005-2018:                                                       *
 *      CERN, Switzerland                                                         *
 *      U. of Victoria, Canada                                                    *
 *      MPI-K Heidelberg, Germany                                                 *
 *      U. of Bonn, Germany                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include "TestOptimization.h"
#include "TMVA/DNN/Architectures/Cpu.h"

#include <iostream>

using namespace TMVA::DNN;
using TMVA::DNN::EOptimizer;

int main()
{
   std::cout << "Testing optimization: (single precision)" << std::endl;

   Real_t momentumSinglePrecision = 0.0;
   std::cout << "Stochastic Gradient Descent: ";
   Double_t error = testOptimization<TCpu<Real_t>>(momentumSinglePrecision, EOptimizer::kSGD, false);
   std::cout << "Mean Absolute error = " << error << std::endl;
   if (error > 1e-3) {
      return 1;
   }

   momentumSinglePrecision = 0.9;
   std::cout << "Stochastic Gradient Descent with momentum: ";
   error = testOptimization<TCpu<Real_t>>(momentumSinglePrecision, EOptimizer::kSGD, false);
   std::cout << "Mean Absolute error = " << error << std::endl;
   if (error > 1e-3) {
      return 1;
   }

   momentumSinglePrecision = 0.0;
   std::cout << "Adam Optimizer: ";
   // Adam doesn't use momentum. Passing this as a parameter just to match the function prototype.
   error = testOptimization<TCpu<Real_t>>(momentumSinglePrecision, EOptimizer::kAdam, false);
   std::cout << "Mean Absolute error = " << error << std::endl;
   if (error > 1e-3) {
      return 1;
   }

   std::cout << std::endl << "Testing optimization: (double precision)" << std::endl;

   Double_t momentumDoublePrecision = 0.0;
   std::cout << "Stochastic Gradient Descent: ";
   error = testOptimization<TCpu<Double_t>>(momentumDoublePrecision, EOptimizer::kSGD, false);
   std::cout << "Mean Absolute error = " << error << std::endl;
   if (error > 1e-5) {
      return 1;
   }

   momentumDoublePrecision = 0.9;
   std::cout << "Stochastic Gradient Descent with momentum: ";
   error = testOptimization<TCpu<Double_t>>(momentumDoublePrecision, EOptimizer::kSGD, false);
   std::cout << "Mean Absolute error = " << error << std::endl;
   if (error > 1e-5) {
      return 1;
   }

   momentumDoublePrecision = 0.0;
   std::cout << "Adam Optimizer: ";

   // Adam doesn't use momentum. Passing this as a parameter just to match the function prototype.
   error = testOptimization<TCpu<Double_t>>(momentumSinglePrecision, EOptimizer::kAdam, false);
   std::cout << "Mean Absolute error = " << error << std::endl;
   if (error > 1e-5) {
      return 1;
   }

   return 0;
}