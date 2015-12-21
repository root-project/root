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

/*! \class TMVA::TActivationTanh
\ingroup TMVA
Tanh activation function for ANN.
*/

#include "TMVA/TActivationTanh.h"

#include "TMVA/TActivation.h"

#include "TMath.h"
#include "TString.h"

#include <iostream>

ClassImp(TMVA::TActivationTanh);

////////////////////////////////////////////////////////////////////////////////
/// a fast tanh approximation

Double_t TMVA::TActivationTanh::fast_tanh(Double_t arg){
   if (arg > 4.97) return 1;
   if (arg < -4.97) return -1;
   float arg2 = arg * arg;
   float a = arg * (135135.0f + arg2 * (17325.0f + arg2 * (378.0f + arg2)));
   float b = 135135.0f + arg2 * (62370.0f + arg2 * (3150.0f + arg2 * 28.0f));
   return a/b;
}

////////////////////////////////////////////////////////////////////////////////
/// evaluate the tanh

Double_t TMVA::TActivationTanh::Eval(Double_t arg)
{
   return fFAST ? fast_tanh(arg) : TMath::TanH(arg);
}

////////////////////////////////////////////////////////////////////////////////
/// evaluate the derivative

Double_t TMVA::TActivationTanh::EvalDerivative(Double_t arg)
{
   Double_t tmp=Eval(arg);
   return ( 1-tmp*tmp);
}

////////////////////////////////////////////////////////////////////////////////
/// get expressions for the tanh and its derivative
/// whatever that may be good for ...

TString TMVA::TActivationTanh::GetExpression()
{
   TString expr = "tanh(x)\t\t (1-tanh()^2)";
   return expr;
}

////////////////////////////////////////////////////////////////////////////////
/// writes the Tanh sigmoid activation function source code

void TMVA::TActivationTanh::MakeFunction( std::ostream& fout, const TString& fncName )
{
   if (fFAST) {
      fout << "double " << fncName << "(double x) const {" << std::endl;
      fout << "   // fast hyperbolic tan approximation" << std::endl;
      fout << "   if (x > 4.97) return 1;" << std::endl;
      fout << "   if (x < -4.97) return -1;" << std::endl;
      fout << "   float x2 = x * x;" << std::endl;
      fout << "   float a = x * (135135.0f + x2 * (17325.0f + x2 * (378.0f + x2)));" << std::endl;
      fout << "   float b = 135135.0f + x2 * (62370.0f + x2 * (3150.0f + x2 * 28.0f));" << std::endl;
      fout << "   return a / b;" << std::endl;
      fout << "}" << std::endl;
   } else {
      fout << "double " << fncName << "(double x) const {" << std::endl;
      fout << "   // hyperbolic tan" << std::endl;
      fout << "   return tanh(x);" << std::endl;
      fout << "}" << std::endl;
   }
}
