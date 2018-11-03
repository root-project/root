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
Radial basis  activation function for ANN.
*/

#include "TMVA/TActivationRadial.h"

#include "TMVA/TActivation.h"

#include "TMath.h"
#include "TString.h"

#include <iostream>


ClassImp(TMVA::TActivationRadial);

////////////////////////////////////////////////////////////////////////////////
/// evaluate gaussian

Double_t TMVA::TActivationRadial::Eval(Double_t arg)
{
   return TMath::Exp(-arg * arg * 0.5);
}

////////////////////////////////////////////////////////////////////////////////
/// evaluate derivative

Double_t TMVA::TActivationRadial::EvalDerivative(Double_t arg)
{
   return -arg*TMath::Exp(-arg * arg * 0.5);
}

////////////////////////////////////////////////////////////////////////////////
/// get expressions for the gaussian and its derivatives

TString TMVA::TActivationRadial::GetExpression()
{
   TString expr = "TMath::Exp(-x^2/2.0)\t\t-x*TMath::Exp(-x^2/2.0)";
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
