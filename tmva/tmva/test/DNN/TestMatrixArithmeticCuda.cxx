// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

////////////////////////////////////////////////////////////////////
// Concrete instantiation of the generic backpropagation test for //
// CUDA architectures.                                            //
////////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cuda.h"
#include "TestMatrixArithmetic.h"

using namespace TMVA::DNN;

int main()
{
   std::cout << "Testing CUDA matrix arithmetic (double):" << std::endl;
   TCudaTensor<Double_t> dummyD(1, 1);

   double error = testMultiplication<TCuda<Double_t>>(10);
   std::cout << "Multiplication:              "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   error = testHadamrdMultiplication<TCuda<Double_t>>(10);
   std::cout << "Hadamrd Multiplication:      "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   error = testSumColumns<TCuda<Double_t>>(1);
   std::cout << "Column Sum:                  "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   error = testConstAdd<TCuda<Double_t>>(1);
   std::cout << "Const Add:                   "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   error = testConstMult<TCuda<Double_t>>(1);
   std::cout << "Const Mult:                  "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   error = testReciprocalElementWise<TCuda<Double_t>>(1);
   std::cout << "Reciprocal ElementWise:      "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   error = testSquareElementWise<TCuda<Double_t>>(1);
   std::cout << "Square ElementWise:          "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   error = testSqrtElementWise<TCuda<Double_t>>(1);
   std::cout << "Sqrt ElementWise:            "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   std::cout << std::endl << "Testing CUDA matrix arithmetic (float):" << std::endl;
   TCudaTensor<Real_t> dummyS(1, 1);

   error = testMultiplication<TCuda<Real_t>>(10);
   std::cout << "Multiplication:              "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;

   error = testSumColumns<TCuda<Real_t>>(1);
   std::cout << "Column Sum:                  "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;

   error = testConstAdd<TCuda<Real_t>>(1);
   std::cout << "Const Add:                   "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;

   error = testConstMult<TCuda<Real_t>>(1);
   std::cout << "Const Mult:                  "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;

   error = testReciprocalElementWise<TCuda<Real_t>>(1);
   std::cout << "Reciprocal ElementWise:      "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;

   error = testSquareElementWise<TCuda<Real_t>>(1);
   std::cout << "Square ElementWise:          "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;

   error = testSqrtElementWise<TCuda<Real_t>>(1);
   std::cout << "Sqrt ElementWise:            "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;
}
