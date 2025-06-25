// @(#)root/tmva $Id$
// Author: Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::TNeuronInputAbs                                                 *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *      TNeuron input calculator -- calculates the sum of the absolute values     *
 *      of the weighted inputs                                                    *
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


#ifndef ROOT_TMVA_TNeuronInputAbs
#define ROOT_TMVA_TNeuronInputAbs

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNeuronInputAbs                                                      //
//                                                                      //
// TNeuron input calculator -- calculates the sum of the absolute       //
// values of the weighted inputs                                        //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMathBase.h"

#include "TMVA/TNeuronInput.h"

#include "TMVA/TNeuron.h"

namespace TMVA {

   class TNeuronInputAbs : public TNeuronInput {

   public:

      TNeuronInputAbs() {}
      virtual ~TNeuronInputAbs() {}

      // calculate the input value for the neuron
      Double_t GetInput( const TNeuron* neuron ) const override {
         if (neuron->IsInputNeuron()) return 0;
         Double_t result = 0;
         for (Int_t i=0; i < neuron->NumPreLinks(); i++)
            result += TMath::Abs(neuron->PreLinkAt(i)->GetWeightedValue());
         return result;
      }

      // name of the class
      TString GetName() override { return "Sum of weighted activations (absolute value)"; }

      ClassDefOverride(TNeuronInputAbs,0); // Calculates the sum of the absolute values of the weighted inputs
   };

} // namespace TMVA

#endif
