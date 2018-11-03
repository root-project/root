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
Sigmoid activation function for TNeuron.
*/

#include "TMVA/TActivationSigmoid.h"

#include "TMVA/TActivation.h"

#include "TMath.h"
#include "TString.h"

#include <iostream>


ClassImp(TMVA::TActivationSigmoid);

////////////////////////////////////////////////////////////////////////////////
/// evaluate the sigmoid

Double_t TMVA::TActivationSigmoid::Eval(Double_t arg)
{
   return 1.0 / (1.0 + TMath::Exp(-arg));
}

////////////////////////////////////////////////////////////////////////////////
/// evaluate the derivative of the sigmoid

Double_t TMVA::TActivationSigmoid::EvalDerivative(Double_t arg)
{
   Double_t tmp = (1.0 + TMath::Exp(-arg));
   return TMath::Exp(-arg) / (tmp * tmp);
}

////////////////////////////////////////////////////////////////////////////////
/// get expressions for the sigmoid and its derivatives

TString TMVA::TActivationSigmoid::GetExpression()
{
   TString expr = "1.0/(1.0+TMath::Exp(-x))\t\tTMath::Exp(-x)/(1.0+TMath::Exp(-x))^2";
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
