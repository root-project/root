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

/*! \class TMVA::TActivationRadial
\ingroup TMVA
Radial basis  activation function for ANN. This really simple implementation
uses TFormula and should probably be replaced with something more
efficient later.
*/

#include "TMVA/TActivationRadial.h"

#include "TMVA/TActivation.h"

#include "TFormula.h"
#include "TMath.h"
#include "TString.h"

#include <iostream>

static const Int_t  UNINITIALIZED = -1;

ClassImp(TMVA::TActivationRadial)

////////////////////////////////////////////////////////////////////////////////
/// constructor for gaussian with center 0, width 1

TMVA::TActivationRadial::TActivationRadial()
{
   fEqn           = new TFormula("Gaussian",   "TMath::Exp(-x^2/2.0)");
   fEqnDerivative = new TFormula("derivative", "-x*TMath::Exp(-x^2/2.0)");
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::TActivationRadial::~TActivationRadial()
{
   if (fEqn != NULL) delete fEqn;
   if (fEqnDerivative != NULL) delete fEqnDerivative;
}

////////////////////////////////////////////////////////////////////////////////
/// evaluate gaussian

Double_t TMVA::TActivationRadial::Eval(Double_t arg)
{
   if (fEqn == NULL) return UNINITIALIZED;
   return fEqn->Eval(arg);
}

////////////////////////////////////////////////////////////////////////////////
/// evaluate derivative

Double_t TMVA::TActivationRadial::EvalDerivative(Double_t arg)
{
   if (fEqnDerivative == NULL) return UNINITIALIZED;
   return fEqnDerivative->Eval(arg);
}

////////////////////////////////////////////////////////////////////////////////
/// get expressions for the gaussian and its derivatives

TString TMVA::TActivationRadial::GetExpression()
{
   TString expr = "";

   if (fEqn == NULL) expr += "<null>";
   else              expr += fEqn->GetExpFormula();

   expr += "\t\t";

   if (fEqnDerivative == NULL) expr += "<null>";
   else                        expr += fEqnDerivative->GetExpFormula();

   return expr;
}

////////////////////////////////////////////////////////////////////////////////
/// writes the sigmoid activation function source code

void TMVA::TActivationRadial::MakeFunction( std::ostream& fout, const TString& fncName )
{
   fout << "double " << fncName << "(double x) const {" << std::endl;
   fout << "   // radial" << std::endl;
   fout << "   return exp(-x*x/2.0);" << std::endl;
   fout << "}" << std::endl;
}
