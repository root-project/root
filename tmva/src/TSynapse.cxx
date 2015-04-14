// @(#)root/tmva $Id$
// Author: Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TSynapse                                                              *
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

#include "TMVA/TSynapse.h"

#ifndef ROOT_TMVA_TNeuron
#include "TMVA/TNeuron.h"
#endif

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif

#include "ThreadLocalStorage.h"

static const Int_t fgUNINITIALIZED = -1;

ClassImp(TMVA::TSynapse);

//______________________________________________________________________________
TMVA::TSynapse::TSynapse()
  : fWeight( 0 ),
    fLearnRate( 0 ),
    fDelta( 0 ),
    fDEDw( 0 ),
    fCount( 0 ),
    fPreNeuron( NULL ),
    fPostNeuron( NULL )
{
   // constructor
   fWeight     = fgUNINITIALIZED;
}


//______________________________________________________________________________
TMVA::TSynapse::~TSynapse()
{
   // destructor
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
   if (fPreNeuron == NULL)
      Log() << kFATAL << "<GetWeightedValue> synapse not connected to neuron" << Endl;

   return (fWeight * fPreNeuron->GetActivationValue());
}

//______________________________________________________________________________
Double_t TMVA::TSynapse::GetWeightedDelta()
{
   // get error field of post-neuron weighted by synapse weight

   if (fPostNeuron == NULL)
      Log() << kFATAL << "<GetWeightedDelta> synapse not connected to neuron" << Endl;

   return fWeight * fPostNeuron->GetDelta();
}

//______________________________________________________________________________
void TMVA::TSynapse::AdjustWeight()
{
   // adjust the weight based on the error field all ready calculated by CalculateDelta
   Double_t wDelta = fDelta / fCount;
   fWeight += -fLearnRate * wDelta;
   InitDelta();
}

//______________________________________________________________________________
void TMVA::TSynapse::CalculateDelta()
{
   // calculate/adjust the error field for this synapse
   fDelta += fPostNeuron->GetDelta() * fPreNeuron->GetActivationValue();
   fCount++;
}

//______________________________________________________________________________
TMVA::MsgLogger& TMVA::TSynapse::Log() const
{
   TTHREAD_TLS_DECL_ARG(MsgLogger,logger,"TSynapse");  //! message logger, static to save resources
   return logger;
}
