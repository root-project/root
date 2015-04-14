// @(#)root/tmva $Id$
// Author: Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TNeuron                                                               *
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
// Neuron class used by TMVA artificial neural network methods
//
//_______________________________________________________________________

#include "TH1D.h"

#ifndef ROOT_TMVA_MsgLogger
#include "TMVA/MsgLogger.h"
#endif
#ifndef ROOT_TMVA_TNeuron
#include "TMVA/TNeuron.h"
#endif
#ifndef ROOT_TMVA_TActivation
#include "TMVA/TActivation.h"
#endif
#ifndef ROOT_TMVA_Tools
#include "TMVA/Tools.h"
#endif
#ifndef ROOT_TMVA_TNeuronInput
#include "TMVA/TNeuronInput.h"
#endif

static const Int_t UNINITIALIZED = -1;

using std::vector;

ClassImp(TMVA::TNeuron)

//______________________________________________________________________________
TMVA::TNeuron::TNeuron()
{
   // standard constructor
   InitNeuron();
}

TMVA::TNeuron::~TNeuron()
{
   // destructor
   if (fLinksIn != NULL)  delete fLinksIn;
   if (fLinksOut != NULL) delete fLinksOut;
}

void TMVA::TNeuron::InitNeuron()
{
   // initialize the neuron, most variables still need to be set via setters
   fLinksIn = new TObjArray();
   fLinksOut = new TObjArray();
   fValue = UNINITIALIZED;
   fActivationValue = UNINITIALIZED;
   fDelta = UNINITIALIZED;
   fDEDw = UNINITIALIZED;
   fError = UNINITIALIZED;
   fActivation = NULL;
   fForcedValue = kFALSE;
   fInputCalculator = NULL;
}

//______________________________________________________________________________
void TMVA::TNeuron::ForceValue(Double_t value)
{
   // force the value, typically for input and bias neurons
   fValue = value;
   fForcedValue = kTRUE;
}

//______________________________________________________________________________
void TMVA::TNeuron::CalculateValue()
{
   // calculate neuron input
   if (fForcedValue) return;
   fValue = fInputCalculator->GetInput(this);
}

//______________________________________________________________________________
void TMVA::TNeuron::CalculateActivationValue()
{
   // calculate neuron activation/output

   if (fActivation == NULL) {
      PrintMessage( kWARNING ,"No activation equation specified." );
      fActivationValue = UNINITIALIZED;
      return;
   }
   fActivationValue = fActivation->Eval(fValue);
}


//______________________________________________________________________________
void TMVA::TNeuron::CalculateDelta()
{
   // calculate error field

   // no need to adjust input neurons
   if (IsInputNeuron()) {
      fDelta = 0.0;
      return;
   }

   Double_t error;


   // output neuron should have error set all ready
   if (IsOutputNeuron()) error = fError;

   // need to calculate error for any other neuron
   else {
      error = 0.0;
      TSynapse* synapse = NULL;
      // Replaced TObjArrayIter pointer by object, as creating it on the stack
      // is much faster (5-10% improvement seen) than re-allocating the new
      // memory for the pointer each time. Thansk to Peter Elmer who pointed this out
      //      TObjArrayIter* iter = (TObjArrayIter*)fLinksOut->MakeIterator();
      TObjArrayIter iter(fLinksOut);
      while (true) {
         synapse = (TSynapse*) iter.Next();
         if (synapse == NULL) break;
         error += synapse->GetWeightedDelta();
      }

   }

   fDelta = error * fActivation->EvalDerivative(GetValue());
}

//______________________________________________________________________________
void TMVA::TNeuron::SetInputCalculator(TNeuronInput* calculator)
{
   // set input calculator
   if (fInputCalculator != NULL) delete fInputCalculator;
   fInputCalculator = calculator;
}

//______________________________________________________________________________
void TMVA::TNeuron::SetActivationEqn(TActivation* activation)
{
   // set activation equation
   if (fActivation != NULL) delete fActivation;
   fActivation = activation;
}

//______________________________________________________________________________
void TMVA::TNeuron::AddPreLink(TSynapse* pre)
{
   // add synapse as a pre-link to this neuron
   if (IsInputNeuron()) return;
   fLinksIn->Add(pre);
}

//______________________________________________________________________________
void TMVA::TNeuron::AddPostLink(TSynapse* post)
{
   // add synapse as a post-link to this neuron
   if (IsOutputNeuron()) return;
   fLinksOut->Add(post);
}

//______________________________________________________________________________
void TMVA::TNeuron::DeletePreLinks()
{
   // delete all pre-links
   DeleteLinksArray(fLinksIn);
}

//______________________________________________________________________________
void TMVA::TNeuron::DeleteLinksArray(TObjArray*& links)
{
   // delete an array of TSynapses

   if (links == NULL) return;

   TSynapse* synapse = NULL;
   Int_t numLinks = links->GetEntriesFast();
   for (Int_t i=0; i<numLinks; i++) {
      synapse = (TSynapse*)links->At(i);
      if (synapse != NULL) delete synapse;
   }
   delete links;
   links = NULL;
}

//______________________________________________________________________________
void TMVA::TNeuron::SetError(Double_t error)
{
   // set error, this should only be done for an output neuron
   if (!IsOutputNeuron())
      PrintMessage( kWARNING, "Warning! Setting an error on a non-output neuron is probably not what you want to do." );

   fError = error;
}

//______________________________________________________________________________
void TMVA::TNeuron::UpdateSynapsesBatch()
{
   // update and adjust the pre-synapses for each neuron (input neuron has no pre-synapse)
   // this method should only be called in batch mode

   if (IsInputNeuron()) return;

   TSynapse* synapse = NULL;
   TObjArrayIter iter(fLinksIn);
   while (true) {
      synapse = (TSynapse*) iter.Next();
      if (synapse == NULL) break;
      synapse->CalculateDelta();
   }

}

//______________________________________________________________________________
void TMVA::TNeuron::UpdateSynapsesSequential()
{
   // update the pre-synapses for each neuron (input neuron has no pre-synapse)
   // this method should only be called in sequential mode

   if (IsInputNeuron()) return;

   TSynapse* synapse = NULL;
   TObjArrayIter iter(fLinksIn);

   while (true) {
      synapse = (TSynapse*) iter.Next();
      if (synapse == NULL) break;
      synapse->InitDelta();
      synapse->CalculateDelta();
      synapse->AdjustWeight();
   }

}

//______________________________________________________________________________
void TMVA::TNeuron::AdjustSynapseWeights()
{
   // adjust the pre-synapses' weights for each neuron (input neuron has no pre-synapse)
   // this method should only be called in batch mode

   if (IsInputNeuron()) return;

   TSynapse* synapse = NULL;
   TObjArrayIter iter(fLinksIn);


   while (true) {
      synapse = (TSynapse*) iter.Next();
      if (synapse == NULL) break;
      synapse->AdjustWeight();
   }

}

//______________________________________________________________________________
void TMVA::TNeuron::InitSynapseDeltas()
{
   // initialize the error fields of all pre-neurons
   // this method should only be called in batch mode

   // an input neuron has no pre-weights to adjust
   if (IsInputNeuron()) return;

   TSynapse* synapse = NULL;
   TObjArrayIter iter(fLinksIn);

   while (true) {
      synapse = (TSynapse*) iter.Next();

      if (synapse == NULL) break;
      synapse->InitDelta();
   }

}

//______________________________________________________________________________
void TMVA::TNeuron::PrintLinks(TObjArray* links) const
{
   // print an array of TSynapses, for debugging

   if (links == NULL) {
      Log() << kDEBUG << "\t\t\t<none>" << Endl;
      return;
   }

   TSynapse* synapse;

   Int_t numLinks = links->GetEntriesFast();
   for  (Int_t i = 0; i < numLinks; i++) {
      synapse = (TSynapse*)links->At(i);
      Log() << kDEBUG <<
         "\t\t\tweighta: " << synapse->GetWeight()
           << "\t\tw-value: " << synapse->GetWeightedValue()
           << "\t\tw-delta: " << synapse->GetWeightedDelta()
           << "\t\tl-rate: "  << synapse->GetLearningRate()
           << Endl;
   }
}

//______________________________________________________________________________
void TMVA::TNeuron::PrintActivationEqn()
{
   // print activation equation, for debugging
   if (fActivation != NULL) Log() << kDEBUG << fActivation->GetExpression() << Endl;
   else                     Log() << kDEBUG << "<none>" << Endl;
}

//______________________________________________________________________________
void TMVA::TNeuron::PrintMessage( EMsgType type, TString message)
{
   // print message, for debugging
   Log() << type << message << Endl;
}

//______________________________________________________________________________
TMVA::MsgLogger& TMVA::TNeuron::Log() const
{
   TTHREAD_TLS_DECL_ARG2(MsgLogger,logger,"TNeuron",kDEBUG);    //! message logger, static to save resources
   return logger;
}
