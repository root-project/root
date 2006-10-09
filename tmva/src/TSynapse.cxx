// @(#)root/tmva $Id: TSynapse.cxx,v 1.14 2006/10/04 22:29:27 andreas.hoecker Exp $
// Author: Matt Jachowski 

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::TSynapse                                                        *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Implementation (see header for description)                               *
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
// Synapse class used by TMVA artificial neural network methods      
//_______________________________________________________________________

#include "Riostream.h"

#ifndef ROOT_TMVA_TSynapse
#include "TMVA/TSynapse.h"
#endif
#ifndef ROOT_TMVA_TNeuron
#include "TMVA/TNeuron.h"
#endif

static const Int_t UNINITIALIZED = -1;

ClassImp(TMVA::TSynapse)

   //______________________________________________________________________________
   TMVA::TSynapse::TSynapse()
{
   // constructor

   fPreNeuron = NULL;
   fPostNeuron = NULL;
   fWeight = UNINITIALIZED;
   fLearnRate = 1.0;
   fCounter = 0;
}

//______________________________________________________________________________
void TMVA::TSynapse::SetWeight(Double_t weight)
{
   // set synapse weight

   fWeight = weight;
}

//______________________________________________________________________________
Double_t TMVA::TSynapse::GetWeightedValue()
{
   // get output of pre-neuron weighted by synapse weight

   if (fPreNeuron == NULL) {
      cout << "Error: Synapse not connected to neuron" << endl;
      exit(1);
   }

   return (fWeight * fPreNeuron->GetActivationValue());
}

//______________________________________________________________________________
Double_t TMVA::TSynapse::GetWeightedDelta()
{
   // get error field of post-neuron weighted by synapse weight

   if (fPostNeuron == NULL) {
      cout << "Error: Synapse not connected to neuron" << endl;
      exit(1);
   }

   return fWeight * fPostNeuron->GetDelta();
}

//______________________________________________________________________________
void TMVA::TSynapse::AdjustWeight()
{
   // adjust the weight based on the error field all ready calculated by CalculateDelta

   Double_t wDelta = fDelta / fCount;
   fWeight += -fLearnRate * wDelta;
}

//______________________________________________________________________________
void TMVA::TSynapse::CalculateDelta()
{
   // calculate/adjust the error field for this synapse

   fDelta += fPostNeuron->GetDelta() * fPreNeuron->GetActivationValue();
   fCount++;
}
