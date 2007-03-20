// @(#)root/tmva $Id: TNeuronInputAbs.h,v 1.6 2006/11/20 15:35:28 brun Exp $
// Author: Matt Jachowski 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::TNeuronInputAbs                                                 *
 * Web    : http://tmva.sourceforge.net                                           *
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
 * (http://tmva.sourceforge.net/LICENSE)                                          *
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

#include "TObject.h"
#include "TString.h"
#include "TMathBase.h"

#ifndef ROOT_TMVA_TNeuronInput
#include "TMVA/TNeuronInput.h"
#endif

#ifndef ROOT_TMVA_TNeuron
#include "TMVA/TNeuron.h"
#endif

namespace TMVA {
  
   class TNeuronInputAbs : public TNeuronInput {
    
   public:

      TNeuronInputAbs() {}
      virtual ~TNeuronInputAbs() {}

      // calculate the input value for the neuron
      Double_t GetInput(TNeuron* neuron) {
         if (neuron->IsInputNeuron()) return 0;
         Double_t result = 0;
         for (Int_t i=0; i < neuron->NumPreLinks(); i++)
            result += TMath::Abs(neuron->PreLinkAt(i)->GetWeightedValue());
         return result;
      }

      // name of the class
      TString GetName() { return "Sum of weighted activations (absolute value)"; }

      ClassDef(TNeuronInputAbs,0) // Calculates the sum of the absolute values of the weighted inputs
         ;
   };

} // namespace TMVA

#endif
