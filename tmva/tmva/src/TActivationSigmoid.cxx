// @(#)root/tmva $Id$
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

/*! \class TMVA::TActivationSigmoid
\ingroup TMVA
Sigmoid activation function for TNeuron. This really simple implementation
uses TFormula and should probably be replaced with something more
efficient later.
*/

#include "TMVA/TActivationSigmoid.h"

#include "TMVA/TActivation.h"

#include "TFormula.h"
#include "TMath.h"
#include "TString.h"

#include <iostream>

static const Int_t  UNINITIALIZED = -1;

ClassImp(TMVA::TActivationSigmoid)

////////////////////////////////////////////////////////////////////////////////
/// constructor for sigmoid normalized in [0,1]

TMVA::TActivationSigmoid::TActivationSigmoid()
{
   fEqn = new TFormula("sigmoid", "1.0/(1.0+TMath::Exp(-x))");
   fEqnDerivative =
      new TFormula("derivative", "TMath::Exp(-x)/(1.0+TMath::Exp(-x))^2");
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::TActivationSigmoid::~TActivationSigmoid()
{
   if (fEqn != NULL) delete fEqn;
   if (fEqnDerivative != NULL) delete fEqnDerivative;
}

////////////////////////////////////////////////////////////////////////////////
/// evaluate the sigmoid

Double_t TMVA::TActivationSigmoid::Eval(Double_t arg)
{
   if (fEqn == NULL) return UNINITIALIZED;
   return fEqn->Eval(arg);

   //return EvalFast(arg);
}

////////////////////////////////////////////////////////////////////////////////
/// evaluate the derivative of the sigmoid

Double_t TMVA::TActivationSigmoid::EvalDerivative(Double_t arg)
{
   if (fEqnDerivative == NULL) return UNINITIALIZED;
   return fEqnDerivative->Eval(arg);

   //return EvalDerivativeFast(arg);
}

////////////////////////////////////////////////////////////////////////////////
/// get expressions for the sigmoid and its derivatives

TString TMVA::TActivationSigmoid::GetExpression()
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

void TMVA::TActivationSigmoid::MakeFunction( std::ostream& fout, const TString& fncName )
{
   fout << "double " << fncName << "(double x) const {" << std::endl;
   fout << "   // sigmoid" << std::endl;
   fout << "   return 1.0/(1.0+exp(-x));" << std::endl;
   fout << "}" << std::endl;
}
