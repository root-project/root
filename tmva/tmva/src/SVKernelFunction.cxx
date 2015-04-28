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
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *      MPI-K Heidelberg, Germany                                                 *
 *      PAN, Krakow, Poland                                                       *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/

#include "TMVA/SVKernelFunction.h"
#include "TMVA/SVEvent.h"
#include "TMath.h"
#include <vector>

//_______________________________________________________________________
TMVA::SVKernelFunction::SVKernelFunction()
   : fGamma(0.),
     fKernel(kRBF),  // kernel, order, theta, and kappa are for backward compatibility
     fOrder(0),
     fTheta(0),
     fKappa(0)
{
   // constructor
}

//_______________________________________________________________________
TMVA::SVKernelFunction::SVKernelFunction( Float_t gamma )
   : fGamma(gamma),
     fKernel(kRBF),  // kernel, order, theta, and kappa are for backward compatibility
     fOrder(0),
     fTheta(0),
     fKappa(0)
{
   // constructor
}

//_______________________________________________________________________
TMVA::SVKernelFunction::~SVKernelFunction() 
{
   // destructor
}

//_______________________________________________________________________
void TMVA::SVKernelFunction::setCompatibilityParams(EKernelType k, UInt_t order, Float_t theta, Float_t kappa) {
   // set old options for compatibility mode
   fKernel = k;
   fOrder = order;
   fTheta = theta;
   fKappa = kappa;
}

//_______________________________________________________________________
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
   case kPolynomial:
      {
         std::vector<Float_t> *v1 = ev1->GetDataVector();
         std::vector<Float_t> *v2 = ev2->GetDataVector();
         Float_t prod = fTheta;
         for (UInt_t idx = 0; idx < v1->size(); idx++) prod += (*v1)[idx] * (*v2)[idx];

         Float_t result = 1.;
         Int_t i = fOrder;
         for (; i > 0; i /= 2) {
            if (i%2) result = prod;
            prod *= prod;
         }
         return result;
      }
   case kLinear:
      {
         std::vector<Float_t> *v1 = ev1->GetDataVector();
         std::vector<Float_t> *v2 = ev2->GetDataVector();
         Float_t prod = 0;
         for (UInt_t i = 0; i < v1->size(); i++) prod += (*v1)[i] * (*v2)[i];
         return prod;
      }
   case kSigmoidal:
      {
         std::vector<Float_t> *v1 = ev1->GetDataVector();
         std::vector<Float_t> *v2 = ev2->GetDataVector();
         Float_t prod = 0;
         for (UInt_t i = 0; i < v1->size(); i++) prod += ((*v1)[i] -(*v2)[i]) *((*v1)[i] -(*v2)[i]) ;
         prod *= fKappa;
         prod += fTheta;
         return TMath::TanH( prod );
      }
   }
   return 0;
}

