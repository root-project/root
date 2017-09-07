// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////
// Test for the loss function implementatoins for the           //
// multi-threaded CPU version using the generic test defined in //
// TestLossFunctions.h.                                         //
//////////////////////////////////////////////////////////////////

#include <iostream>
#include "TMVA/DNN/Architectures/Cpu.h"
#include "TestLossFunctions.h"

using namespace TMVA::DNN;

int main()
{
   using Scalar_t = Double_t;

   std::cout << "Testing Loss Functions:" << std::endl << std::endl;

   double error;

   //
   // Mean Squared Error.
   //

   error = testMeanSquaredError<TCpu<Scalar_t>>(10);
   std::cout << "Testing mean squared error loss:        ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-3) return 1;

   error = testMeanSquaredErrorGradients<TCpu<Scalar_t>>(10);
   std::cout << "Testing mean squared error gradient:    ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-3) return 1;

   //
   // Cross Entropy.
   //

   error = testCrossEntropy<TCpu<Scalar_t>>(10);
   std::cout << "Testing cross entropy loss:             ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-3) return 1;

   error = testCrossEntropyGradients<TCpu<Scalar_t>>(10);
   std::cout << "Testing mean squared error gradient:    ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-3) return 1;

   //
   // Softmax Cross Entropy.
   //

   error = testSoftmaxCrossEntropy<TCpu<Scalar_t>>(10);
   std::cout << "Testing softmax cross entropy loss:     ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-3) return 1;

   error = testSoftmaxCrossEntropyGradients<TCpu<Scalar_t>>(10);
   std::cout << "Testing softmax cross entropy gradient: ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-3) return 1;
}
