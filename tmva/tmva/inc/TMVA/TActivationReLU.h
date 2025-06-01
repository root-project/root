// @(#)root/tmva $Id$
// Author: Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::TActivationReLU                                                 *
 *                                             *
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
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_TActivationReLU
#define ROOT_TMVA_TActivationReLU

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TActivationReLU                                                      //
//                                                                      //
// Rectified Linear Unit activation function for TNeuron                //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TString.h"

#include "TMVA/TActivation.h"

namespace TMVA {

   class TActivationReLU : public TActivation {

   public:
      TActivationReLU() {}
      ~TActivationReLU() {}

      // evaluate the activation function
      Double_t Eval(Double_t arg) override { return arg>0 ? arg : 0;}

      // evaluate the derivative of the activation function
      Double_t EvalDerivative(Double_t arg) override { return arg>0 ? 1 : 0;}

      // minimum of the range of the activation function
      Double_t GetMin() override { return -1; }

      // maximum of the range of the activation function
      Double_t GetMax() override { return 1; }

      // expression for the activation function
      TString GetExpression() override;

      // writer of function code
      void MakeFunction(std::ostream& fout, const TString& fncName) override;

   private:
      ClassDefOverride(TActivationReLU, 0); // Rectified Linear Unit activation function for TNeuron
   };

} // namespace TMVA

#endif
