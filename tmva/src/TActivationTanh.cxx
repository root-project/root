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


ClassImp(TMVA::TActivationTanh)

//______________________________________________________________________________
TMVA::TActivationTanh::TActivationTanh()
{
   // constructor for tanh sigmoid (normalized in [-1,1])
   fFAST=kTRUE;
}

//______________________________________________________________________________
TMVA::TActivationTanh::~TActivationTanh()
{
   // destructor
}

//______________________________________________________________________________
Double_t TMVA::TActivationTanh::fast_tanh(Double_t arg){
   // a fast tanh approximation

   float arg2 = arg * arg;
   float a = arg * (135135.0f + arg2 * (17325.0f + arg2 * (378.0f + arg2)));
   float b = 135135.0f + arg2 * (62370.0f + arg2 * (3150.0f + arg2 * 28.0f));
   return a / b;
}

//______________________________________________________________________________
Double_t TMVA::TActivationTanh::Eval(Double_t arg)
{
   // evaluate the tanh

   return fFAST ? fast_tanh(arg) : TMath::TanH(arg);
}

//______________________________________________________________________________
Double_t TMVA::TActivationTanh::EvalDerivative(Double_t arg)
{
   // evaluate the derivative

   Double_t tmp=Eval(arg);
   return ( 1-tmp*tmp);
}

//______________________________________________________________________________
TString TMVA::TActivationTanh::GetExpression()
{
   // get expressions for the tanh and its derivative
   // whatever that may be good for ... 

   TString expr = "tanh(x)\t\t (1-tanh()^2)";
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
