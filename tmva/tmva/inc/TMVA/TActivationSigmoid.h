// @(#)root/tmva $Id$
// Author: Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::TActivationSigmoid                                              *
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

#ifndef ROOT_TMVA_TActivationSigmoid
#define ROOT_TMVA_TActivationSigmoid

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TActivationSigmoid                                                   //
//                                                                      //
// Sigmoid activation function for TNeuron                              //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TString.h"

#include "TMVA/TActivation.h"

namespace TMVA {

   class TActivationSigmoid : public TActivation {

   public:
      TActivationSigmoid() {}
      ~TActivationSigmoid() {}

      // evaluate the activation function
      Double_t Eval(Double_t arg);

      // evaluate the derivative of the activation function
      Double_t EvalDerivative(Double_t arg);

      // minimum of the range of the activation function
      Double_t GetMin() { return 0; }

      // maximum of the range of the activation function
      Double_t GetMax() { return 1; }

      // expression for the activation function
      TString GetExpression();

      // writer of function code
      virtual void MakeFunction(std::ostream& fout, const TString& fncName);

   private:

      ClassDef(TActivationSigmoid,0); // Sigmoid activation function for TNeuron
   };

} // namespace TMVA

#endif
