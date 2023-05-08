// @(#)root/tmva $Id$
// Author: Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::TActivationTanh                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Tanh activation function for TNeuron                                      *
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

#ifndef ROOT_TMVA_TActivationTanh
#define ROOT_TMVA_TActivationTanh

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TActivationTanh                                                      //
//                                                                      //
// Tanh activation function for TNeuron                                 //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TString.h"

#include "TMVA/TActivation.h"

namespace TMVA {

   class TActivationTanh : public TActivation {

   public:

      TActivationTanh() {}
      ~TActivationTanh() {}

      // evaluate the activation function
      Double_t Eval(Double_t arg);

      // evaluate the derivative of the activation function
      Double_t EvalDerivative(Double_t arg);

      // minimum of the range of the activation function
      Double_t GetMin() { return -1; }

      // maximum of the range of the activation function
      Double_t GetMax() { return 1; }

      // expression for the activation function
      TString GetExpression();

      // writer of function code
      virtual void MakeFunction(std::ostream& fout, const TString& fncName);

      void SetSlow(){fFAST=kFALSE;} // to ensure old training files will be process with old tanh code
   private:
      // fast tanh approximation
      Double_t fast_tanh(Double_t arg);
      Bool_t   fFAST;

      ClassDef(TActivationTanh,0); // Tanh sigmoid activation function for TNeuron
   };

} // namespace TMVA

#endif
