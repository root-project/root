// @(#)root/tmva $Id$
// Author: Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::TActivationIdentity                                             *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Identity activation function for TNeuron                                  *
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

#ifndef ROOT_TMVA_TActivationIdentity
#define ROOT_TMVA_TActivationIdentity

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TActivationIdentity                                                  //
//                                                                      //
// Identity activation function for TNeuron                             //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TString.h"

#include "TMVA/TActivation.h"

namespace TMVA {

   class TActivationIdentity : public TActivation {

   public:

      TActivationIdentity() {}
      ~TActivationIdentity() {}

      // evaluate the activation function
      virtual Double_t Eval(Double_t arg) { return arg; } // f(x) = x

      // evaluate the derivative of the activation function
      virtual Double_t EvalDerivative(Double_t) {
         return 1; // f'(x) = 1
      }

      // minimum of the range of the activation function
      virtual Double_t GetMin() { return 0; } // these should never be called

      // maximum of the range of the activation function
      virtual Double_t GetMax() { return 1; } // these should never be called

      // expression for activation function
      virtual TString GetExpression() { return "x\t1"; }

      // writer of function code
      virtual void MakeFunction(std::ostream& fout, const TString& fncName);

   private:

      ClassDef(TActivationIdentity,0); // Identity activation function for TNeuron
   };

} // namespace TMVA

#endif
