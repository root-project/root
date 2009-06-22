// @(#)root/tmva $Id$
// Author: Matt Jachowski 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TActivationRadial                                                     *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Radial basis activation function for TNeuron                              *
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
//  Radial basis  activation function for ANN. This really simple implementation
//  uses TFormulas and should probably be replaced with something more
//  efficient later.
//_______________________________________________________________________

#include <iostream>

#include "TFormula.h"
#include "TString.h"
#include "TMath.h"

#ifndef ROOT_TMVA_TActivationRadial
#include "TMVA/TActivationRadial.h"
#endif

static const Int_t  UNINITIALIZED = -1;

ClassImp(TMVA::TActivationRadial)

//______________________________________________________________________________
TMVA::TActivationRadial::TActivationRadial()
{
   // constructor for gaussian with center 0, width 1

   fEqn           = new TFormula("Gaussian",   "TMath::Exp(-x^2/2.0)");
   fEqnDerivative = new TFormula("derivative", "-x*TMath::Exp(-x^2/2.0)");
}

//______________________________________________________________________________
TMVA::TActivationRadial::~TActivationRadial()
{
   // destructor

   if (fEqn != NULL) delete fEqn;
   if (fEqnDerivative != NULL) delete fEqnDerivative;
}

//______________________________________________________________________________
Double_t TMVA::TActivationRadial::Eval(Double_t arg)
{
   // evaluate gaussian

   if (fEqn == NULL) return UNINITIALIZED;
   return fEqn->Eval(arg);
}

//______________________________________________________________________________
Double_t TMVA::TActivationRadial::EvalDerivative(Double_t arg)
{
   // evaluate derivative

   if (fEqnDerivative == NULL) return UNINITIALIZED;
   return fEqnDerivative->Eval(arg);
}

//______________________________________________________________________________
TString TMVA::TActivationRadial::GetExpression()
{
   // get expressions for the gaussian and its derivatives

   TString expr = "";

   if (fEqn == NULL) expr += "<null>";
   else              expr += fEqn->GetExpFormula();

   expr += "\t\t";

   if (fEqnDerivative == NULL) expr += "<null>";
   else                        expr += fEqnDerivative->GetExpFormula();

   return expr;
}

//______________________________________________________________________________
void TMVA::TActivationRadial::MakeFunction( std::ostream& fout, const TString& fncName ) 
{
   // writes the sigmoid activation function source code
   fout << "double " << fncName << "(double x) const {" << std::endl;
   fout << "   // radial" << std::endl;
   fout << "   return exp(-x*x/2.0);" << std::endl;
   fout << "}" << std::endl;
}
