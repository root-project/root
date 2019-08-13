// @(#)root/tmva $Id$
// Author: Simon Pfreundschuh

/*************************************************************************
 * Copyright (C) 2016, Simon Pfreundschuh                                *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//////////////////////////////////////////////////////////////////////
//  Concrete instantiation of the generic activation function test  //
//  for the multi-threaded CPU implementation.                      //
//////////////////////////////////////////////////////////////////////

#include <iostream>
#include  "RConfigure.h"
#include "TMVA/DNN/Architectures/Cpu.h"
#include "Utility.h"
#include "TestActivationFunctions.h"

using namespace TMVA::DNN;

int main()
{
   using Scalar_t = Double_t;

   std::cout << "Testing Activation Functions:" << std::endl;

   double error;

   // Identity.

   error = testIdentity<TCpu<Scalar_t>>(10);
   std::cout << "Testing identity activation:            ";
   std::cout << "maximum relative error = " << error << std::endl;
   if (error > 1e-10)
       return 1;

#if 0   // disable now (uses TReference)
   error = testIdentityDerivative<TCpu<Scalar_t>>(10);
   std::cout << "Testing identity activation derivative: ";
   std::cout << "maximum relative error = " << error << std::endl;
   if (error > 1e-10)
       return 1;
#endif

   // ReLU.

   error = testRelu<TCpu<Scalar_t>>(10);
   std::cout << "Testing ReLU activation:                ";
   std::cout << "maximum relative error = " << error << std::endl;
   if (error > 1e-10)
       return 1;

   error = testReluDerivative<TCpu<Scalar_t>>(10);
   std::cout << "Testing ReLU activation derivative:     ";
   std::cout << "maximum relative error = " << error << std::endl;
   if (error > 1e-10)
       return 1;

   // Sigmoid.

   error = testSigmoid<TCpu<Scalar_t>>(10);
   std::cout << "Testing Sigmoid activation:             ";
   std::cout << "maximum relative error = " << error << std::endl;
   if (error > 1e-10)
       return 1;

   error = testSigmoidDerivative<TCpu<Scalar_t>>(10);
   std::cout << "Testing Sigmoid activation derivative:  ";
   std::cout << "maximum relative error = " << error << std::endl;
   if (error > 1e-10)
       return 1;

   // TanH.

   error = testTanh<TCpu<Scalar_t>>(10);
   std::cout << "Testing TanH activation:                   ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
#ifdef R__HAS_VDT   // error is larger when using fast tanh from vdt
    if (error > 1e-6) 
#else
    if (error > 1e-10)
#endif
       return 1;

   error = testTanhDerivative<TCpu<Scalar_t>>(10);
   std::cout << "Testing TanH activation derivative:        ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
#ifdef R__HAS_VDT   // error is larger when using fast tanh from vdt
    if (error > 1e-3) 
#else
    if (error > 1e-10)
#endif
       return 1;

   // Symmetric ReLU.

   error = testSymmetricRelu<TCpu<Scalar_t>>(10);
   std::cout << "Testing Symm. ReLU activation:             ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-10)
       return 1;

   error = testSymmetricReluDerivative<TCpu<Scalar_t>>(10);
   std::cout << "Testing Symm. ReLU activation derivative:  ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-10)
       return 1;

   // Soft Sign.

   error = testSoftSign<TCpu<Scalar_t>>(10);
   std::cout << "Testing Soft Sign activation:              ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-10)
       return 1;

   error = testSoftSignDerivative<TCpu<Scalar_t>>(10);
   std::cout << "Testing Soft Sign activation derivative:   ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-10)
       return 1;

   // Gauss.

   error = testGauss<TCpu<Scalar_t>>(10);
   std::cout << "Testing Gauss activation:                  ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-10)
       return 1;

   error = testGaussDerivative<TCpu<Scalar_t>>(10);
   std::cout << "Testing Gauss activation derivative:       ";
   std::cout << "maximum relative error = " << print_error(error) << std::endl;
   if (error > 1e-10)
       return 1;

   return 0;
}
