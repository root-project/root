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
   std::cout << "Multiplication:              "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   error = testHadamrdMultiplication<TCpu<Double_t>>(10);
   std::cout << "Hadamrd Multiplication:      "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   error = testSumColumns<TCpu<Double_t>>(1);
   std::cout << "Column Sum:                  "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   error = testConstAdd<TCpu<Double_t>>(1);
   std::cout << "Const Add:                   "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   error = testConstMult<TCpu<Double_t>>(1);
   std::cout << "Const Mult:                  "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   error = testReciprocalElementWise<TCpu<Double_t>>(1);
   std::cout << "Reciprocal ElementWise:      "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   error = testSquareElementWise<TCpu<Double_t>>(1);
   std::cout << "Square ElementWise:          "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   error = testSqrtElementWise<TCpu<Double_t>>(1);
   std::cout << "Sqrt ElementWise:            "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   std::cout << std::endl << "Testing CPU matrix arithmetic (float):" << std::endl;

   error = testMultiplication<TCpu<Real_t>>(10);
   std::cout << "Multiplication:              "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;

   error = testSumColumns<TCpu<Real_t>>(1);
   std::cout << "Column Sum:                  "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;

   error = testConstAdd<TCpu<Real_t>>(1);
   std::cout << "Const Add:                   "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;

   error = testConstMult<TCpu<Real_t>>(1);
   std::cout << "Const Mult:                  "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;

   error = testReciprocalElementWise<TCpu<Real_t>>(1);
   std::cout << "Reciprocal ElementWise:      "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;

   error = testSquareElementWise<TCpu<Real_t>>(1);
   std::cout << "Square ElementWise:          "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;

   error = testSqrtElementWise<TCpu<Real_t>>(1);
   std::cout << "Sqrt ElementWise:            "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;

   return 0;
}
