// @(#)root/tmva $Id$
// Author: Matt Jachowski 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TActivationTanh                                                       *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Tanh activation function (sigmoid normalized in [-1,1] for an ANN.        *
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
//  Tanh activation function for ANN. This really simple implementation
//  uses TFormulas and should probably be replaced with something more
//  efficient later.
//                                                                      
//_______________________________________________________________________

#include <iostream>

#include "TFormula.h"
#include "TString.h"
#include "TMath.h"

#ifndef ROOT_TMVA_TActivationTanh
#include "TMVA/TActivationTanh.h"
#endif

static const Int_t  UNINITIALIZED = -1;

ClassImp(TMVA::TActivationTanh)

//______________________________________________________________________________
TMVA::TActivationTanh::TActivationTanh()
{
   // constructor for tanh sigmoid (normalized in [-1,1])

   fEqn = new TFormula("sigmoid", "TMath::TanH(x)");
   fEqnDerivative = 
      new TFormula("derivative", "1-(TMath::TanH(x))^2");
}

//______________________________________________________________________________
TMVA::TActivationTanh::~TActivationTanh()
{
   // destructor

   if (fEqn != NULL) delete fEqn;
   if (fEqnDerivative != NULL) delete fEqnDerivative;
}

//______________________________________________________________________________
Double_t TMVA::TActivationTanh::Eval(Double_t arg)
{
   // evaluate the tanh

   if (fEqn == NULL) return UNINITIALIZED;
   return fEqn->Eval(arg);
}

//______________________________________________________________________________
Double_t TMVA::TActivationTanh::EvalDerivative(Double_t arg)
{
   // evaluate the derivative

   if (fEqnDerivative == NULL) return UNINITIALIZED;
   return fEqnDerivative->Eval(arg);
}

//______________________________________________________________________________
TString TMVA::TActivationTanh::GetExpression()
{
   // get expressions for the tanh and its derivative

   TString expr = "";

   if (fEqn == NULL) expr += "<null>";
   else              expr += fEqn->GetExpFormula();

   expr += "\t\t";

   if (fEqnDerivative == NULL) expr += "<null>";
   else                        expr += fEqnDerivative->GetExpFormula();

   return expr;
}

//______________________________________________________________________________
void TMVA::TActivationTanh::MakeFunction( std::ostream& fout, const TString& fncName ) 
{
   // writes the sigmoid activation function source code
   fout << "double " << fncName << "(double x) const {" << std::endl;
   fout << "   // hyperbolic tan" << std::endl;
   fout << "   return tanh(x);" << std::endl;
   fout << "}" << std::endl;
}
