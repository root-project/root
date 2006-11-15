// @(#)root/tmva $Id: TActivationSigmoid.cxx,v 1.12 2006/10/10 17:43:52 andreas.hoecker Exp $
// Author: Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TActivationSigmoid                                                    *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Sigmoid activation function for TNeuron                                   *
 *                                                                                *
 * Authors (alphabetical):                                                        *
 *      Matt Jachowski  <jachowski@stanford.edu> - Stanford University, USA       *
 *                                                                                *
 * Copyright (c) 2005:                                                            *
 *      CERN, Switzerland                                                         *
 *                                                                                *
 * Redistribution and use in source and binary forms, with or without             *
 * modification, are permitted according to the terms listed in LICENSE           *
 * (http://tmva.sourceforge.net/LICENSE)                                          *
 **********************************************************************************/
  
//_______________________________________________________________________
//                                                                      
//  Sigmoid activation function for TNeuron. This really simple implementation
//  uses TFormulas and should probably be replaced with something more
//  efficient later.
//                                                                      
//_______________________________________________________________________

#include "TFormula.h"
#include "TString.h"
#include "TMath.h"
#include "Riostream.h"

#ifndef ROOT_TMVA_TActivationSigmoid
#include "TMVA/TActivationSigmoid.h"
#endif

static const Int_t  UNINITIALIZED = -1;

ClassImp(TMVA::TActivationSigmoid)
   ;

//______________________________________________________________________________
TMVA::TActivationSigmoid::TActivationSigmoid()
{
   // constructor for sigmoid normalized in [0,1]
   
   fEqn = new TFormula("sigmoid", "1.0/(1.0+TMath::Exp(-x))");
   fEqnDerivative = 
      new TFormula("derivative", "TMath::Exp(-x)/(1.0+TMath::Exp(-x))^2");
}

//______________________________________________________________________________
TMVA::TActivationSigmoid::~TActivationSigmoid()
{
   // destructor
   
   if (fEqn != NULL) delete fEqn;
   if (fEqnDerivative != NULL) delete fEqnDerivative;
}

//______________________________________________________________________________
Double_t TMVA::TActivationSigmoid::Eval(Double_t arg)
{
   // evaluate the sigmoid

   if (fEqn == NULL) return UNINITIALIZED;
   return fEqn->Eval(arg);

   //return EvalFast(arg);
}

//______________________________________________________________________________
Double_t TMVA::TActivationSigmoid::EvalDerivative(Double_t arg)
{
   // evaluate the derivative of the sigmoid

   if (fEqnDerivative == NULL) return UNINITIALIZED;
   return fEqnDerivative->Eval(arg);

   //return EvalDerivativeFast(arg);
}

//______________________________________________________________________________
TString TMVA::TActivationSigmoid::GetExpression()
{
   // get expressions for the sigmoid and its derivatives
   
   TString expr = "";
   
   if (fEqn == NULL) expr += "<null>";
   else              expr += fEqn->GetExpFormula();
   
   expr += "\t\t";
   
   if (fEqnDerivative == NULL) expr += "<null>";
   else                        expr += fEqnDerivative->GetExpFormula();
   
   return expr;
}
