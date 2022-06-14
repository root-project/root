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
#include "TMVA/DNN/Architectures/Cudnn.h"
#include "TestMatrixArithmetic.h"

using namespace TMVA::DNN;

int main()
{
   std::cout << "Testing CuDNN tesor arithmetic (double):" << std::endl;

   Double_t error = testMultiplication<TCudnn<Double_t>>(10);
   std::cout << "Multiplication:              "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   /*error = testSumColumns<TCudnn<Double_t>>(1);
   std::cout << "Column Sum:                  "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;*/

   error = testConstAdd<TCudnn<Double_t>>(1);
   std::cout << "Const Add:                   "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   error = testConstMult<TCudnn<Double_t>>(1);
   std::cout << "Const Mult:                  "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   /*error = testReciprocalElementWise<TCudnn<Double_t>>(1);
   std::cout << "Reciprocal ElementWise:      "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;*/

   /*error = testSquareElementWise<TCudnn<Double_t>>(1);
   std::cout << "Square ElementWise:          "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;*/

   error = testSqrtElementWise<TCudnn<Double_t>>(1);
   std::cout << "Sqrt ElementWise:            "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-3)
      return 1;

   std::cout << std::endl << "Testing CuDNN tesor arithmetic (float):" << std::endl;

   error = testMultiplication<TCudnn<Real_t>>(10);
   std::cout << "Multiplication:              "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;

   /*error = testSumColumns<TCudnn<Real_t>>(1);
   std::cout << "Column Sum:                  "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;*/

   error = testConstAdd<TCudnn<Real_t>>(1);
   std::cout << "Const Add:                   "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;

   error = testConstMult<TCudnn<Real_t>>(1);
   std::cout << "Const Mult:                  "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;

   /*error = testReciprocalElementWise<TCudnn<Real_t>>(1);
   std::cout << "Reciprocal ElementWise:      "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;*/

   /*error = testSquareElementWise<TCudnn<Real_t>>(1);
   std::cout << "Square ElementWise:          "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;*/

   error = testSqrtElementWise<TCudnn<Real_t>>(1);
   std::cout << "Sqrt ElementWise:            "
             << "Max. rel. error: " << error << std::endl;
   if (error > 1e-1)
      return 1;
}
