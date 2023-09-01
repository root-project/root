// @(#)root/tmva $Id$
// Author: Andrzej Zemla

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : SVKernelFunction                                                      *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation                                                            *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Marcin Wolter  <Marcin.Wolter@cern.ch> - IFJ PAN, Krakow, Poland          *
 *      Andrzej Zemla  <azemla@cern.ch>        - IFJ PAN, Krakow, Poland          *
 *      (IFJ PAN: Henryk Niewodniczanski Inst. Nucl. Physics, Krakow, Poland)     *
 *                                                                                *
 * MultiGaussian, Product and Sum kernels included:                               *
 *      Adrian Bevan   <adrian.bevan@cern.ch>  -         Queen Mary               *
 *                                                       University of London, UK *
 *      Tom Stevenson <thomas.james.stevenson@cern.ch> - Queen Mary               *
 *                                                       University of London, UK *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *      PAN, Krakow, Poland                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

/*! \class TMVA::SVKernelFunction
\ingroup TMVA
Kernel for Support Vector Machine
*/

#include "TMVA/SVKernelFunction.h"
#include "TMVA/SVEvent.h"
#include "TMath.h"
#include <vector>

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::SVKernelFunction::SVKernelFunction()
   : fGamma(0.),
     fKernel(kRBF),  // kernel, order, theta, and kappa are for backward compatibility
     fOrder(0),
     fTheta(0),
     fKappa(0)
{
}

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::SVKernelFunction::SVKernelFunction( Float_t gamma )
   : fGamma(gamma),
     fKernel(kRBF),  // kernel, order, theta, and kappa are for backward compatibility
     fOrder(0),
     fTheta(0),
     fKappa(0)
{
   fmGamma.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::SVKernelFunction::SVKernelFunction( EKernelType k, Float_t param1, Float_t param2)
   :  fKernel(k)
{
   if      (k==kRBF)        { fGamma = param1; }
   else if (k==kPolynomial){
      fOrder = param1;
      fTheta = param2;
   }
   fKernelsList.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::SVKernelFunction::SVKernelFunction( std::vector<float> params ) :
   fKernel(kMultiGauss)
{
   fmGamma.clear();
   for( std::vector<float>::const_iterator iter = params.begin(); iter != params.end()\
           ; ++iter ){
      fmGamma.push_back(*iter);
   }
   //fKernelsList.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// constructor

TMVA::SVKernelFunction::SVKernelFunction(EKernelType k, std::vector<EKernelType> kernels, std::vector<Float_t> gammas, Float_t gamma, Float_t order, Float_t theta) :
   fGamma(gamma),
   fKernel(k),
   fOrder(order),
   fTheta(theta)
{
   fmGamma.clear();
   fKernelsList.clear();
   fKernelsList = kernels;
   fmGamma = gammas;
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::SVKernelFunction::~SVKernelFunction()
{
   fmGamma.clear();
   fKernelsList.clear();
}

////////////////////////////////////////////////////////////////////////////////
/// set old options for compatibility mode

void TMVA::SVKernelFunction::setCompatibilityParams(EKernelType k, UInt_t order, Float_t theta, Float_t kappa) {
   fKernel = k;
   fOrder = order;
   fTheta = theta;
   fKappa = kappa;
}

////////////////////////////////////////////////////////////////////////////////

Float_t TMVA::SVKernelFunction::Evaluate( SVEvent* ev1, SVEvent* ev2 )
{
   switch(fKernel) {
   case kRBF:
      {
         std::vector<Float_t> *v1 = ev1->GetDataVector();
         std::vector<Float_t> *v2 = ev2->GetDataVector();

         Float_t norm = 0;
         for (UInt_t i = 0; i < v1->size(); i++) norm += ((*v1)[i] -(*v2)[i]) *((*v1)[i] -(*v2)[i]) ;

         return TMath::Exp(-norm*fGamma);
      }
   case kMultiGauss:
      {
         // Kernel function with a kernel parameter gamma for
         // each input variable. Described in "An Introduction to
         // Support Vector Machines and Other Kernel-based Learning
         // Methods" by Cristianini and Shawe-Taylor, Section 3.5
         std::vector<Float_t> *v1 = ev1->GetDataVector();
         std::vector<Float_t> *v2 = ev2->GetDataVector();
         if(fmGamma.size() != v1->size()){
            std::cout <<  "Fewer gammas than input variables! #Gammas= " << fmGamma.size() << " #Input variables= " << v1->size() << std::endl;
            std::cout << "***> abort program execution" << std::endl;
            exit(1);
         }

         Float_t result = 1.;
         for (UInt_t i = 0; i < v1->size(); i++) {
            result *= TMath::Exp( -((*v1)[i] -(*v2)[i])*((*v1)[i] -(*v2)[i])*fmGamma[i] );
         }
         return result;
      }
   case kPolynomial:
      {
         // Polynomial kernel of form (z.x + theta)^n
         // it should be noted that the power is currently only integer
         std::vector<Float_t> *v1 = ev1->GetDataVector();
         std::vector<Float_t> *v2 = ev2->GetDataVector();
         Float_t prod = fTheta;
         for (UInt_t idx = 0; idx < v1->size(); idx++) prod += (*v1)[idx] * (*v2)[idx];

         Float_t result = 1.;
         Int_t i = fOrder;
         result = TMath::Power(prod,i);
         return result;
      }
   case kLinear:
      {
         // This is legacy code. The linear polynomial is a special case
         // of the polynomial with order=1 and theta=0.
         std::vector<Float_t> *v1 = ev1->GetDataVector();
         std::vector<Float_t> *v2 = ev2->GetDataVector();
         Float_t prod = 0;
         for (UInt_t i = 0; i < v1->size(); i++) prod += (*v1)[i] * (*v2)[i];
         return prod;
      }
   case kSigmoidal:
      {
         // This kernel doesn't always result in a positive-semidefinite Gram
         // matrix so should be used with caution and therefore not
         // currently accessible. This is not a valid Mercer kernel
         std::vector<Float_t> *v1 = ev1->GetDataVector();
         std::vector<Float_t> *v2 = ev2->GetDataVector();
         Float_t prod = 0;
         for (UInt_t i = 0; i < v1->size(); i++) prod += ((*v1)[i] -(*v2)[i]) *((*v1)[i] -(*v2)[i]) ;
         prod *= fKappa;
         prod += fTheta;
         return TMath::TanH( prod );
      }
   case kProd:
      {
         // Calculate product of kernels by looping over list of kernels
         // and evaluating the value for each, setting kernel back to
         // kProd before returning so it can be used again. Described in "An Introduction to         // Support Vector Machines and Other Kernel-based Learning
         // Methods" by Cristianini and Shawe-Taylor, Section 3.3.2
         Float_t kernelVal;
         kernelVal = 1;
         for(UInt_t i = 0; i<fKernelsList.size(); i++){
            fKernel = fKernelsList.at(i);
            Float_t a = Evaluate(ev1,ev2);
            kernelVal *= a;
         }
         fKernel = kProd;
         return kernelVal;
      }
   case kSum:
      {
         // Calculate sum of kernels by looping over list of kernels
         // and evaluating the value for each, setting kernel back to
         // kSum before returning so it can be used again. Described in "An Introduction to          // Support Vector Machines and Other Kernel-based Learning
         // Methods" by Cristianini and Shawe-Taylor, Section 3.3.2
         Float_t kernelVal = 0;
         for(UInt_t i = 0; i<fKernelsList.size(); i++){
            fKernel = fKernelsList.at(i);
            Float_t a = Evaluate(ev1,ev2);
            kernelVal += a;
         }
         fKernel = kSum;
         return kernelVal;
      }
   }
   return 0;
}

