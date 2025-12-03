// @(#)root/tmva $Id$
// Author: Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::TNeuronInputSqSum                                               *
 *                                             *
 *                                                                                *
 * Description:                                                                   *
 *       TNeuron input calculator -- calculates the square                        *
 *       of the weighted sum of inputs.                                           *
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


#ifndef ROOT_TMVA_TNeuronInputSqSum
#define ROOT_TMVA_TNeuronInputSqSum

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNeuronInputSqSum                                                    //
//                                                                      //
// TNeuron input calculator -- calculates the squared weighted sum of   //
// inputs                                                               //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include "TMVA/TNeuronInput.h"
#include "TMVA/TNeuron.h"

namespace TMVA {

   class TNeuronInputSqSum : public TNeuronInput {

   public:

      TNeuronInputSqSum() {}
      virtual ~TNeuronInputSqSum() {}

      // calculate the input value for the neuron
      Double_t GetInput( const TNeuron* neuron ) const override {
         if (neuron->IsInputNeuron()) return 0;
         Double_t result = 0;
         for (Int_t i=0; i < neuron->NumPreLinks(); i++) {
            Double_t val = neuron->PreLinkAt(i)->GetWeightedValue();
            result += val*val;
         }
         return result;
      }

      // name of the class
      TString GetName() override { return "Sum of weighted activations squared"; }

      ClassDefOverride(TNeuronInputSqSum,0); // Calculates square of  weighted sum of neuron inputs
   };

} // namespace TMVA

#endif
