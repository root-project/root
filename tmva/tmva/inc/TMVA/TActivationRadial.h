// @(#)root/tmva $Id$
// Author: Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::TActivationRadial                                               *
 *                                             *
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
 * (see tmva/doc/LICENSE)                                          *
 **********************************************************************************/

#ifndef ROOT_TMVA_TActivationRadial
#define ROOT_TMVA_TActivationRadial

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TActivationRadial                                                    //
//                                                                      //
// Radial basis activation function for TNeuron                         //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TString.h"

#include "TMVA/TActivation.h"

namespace TMVA {

   class TActivationRadial : public TActivation {

   public:
      TActivationRadial() {}
      ~TActivationRadial() {}

      // evaluate the activation function
      Double_t Eval(Double_t arg) override;

      // evaluate the derivative of the activation function
      Double_t EvalDerivative(Double_t arg) override;

      // minimum of the range of the activation function
      Double_t GetMin() override { return 0; }

      // maximum of the range of the activation function
      Double_t GetMax() override { return 1; }

      // expression for the activation function
      TString GetExpression() override;

      // writer of function code
      void MakeFunction(std::ostream& fout, const TString& fncName) override;

   private:

      ClassDefOverride(TActivationRadial,0);  // Radial basis activation function for TNeuron
   };

} // namespace TMVA

#endif
