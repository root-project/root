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

/*! \class TMVA::TNeuron
\ingroup TMVA
Neuron class used by TMVA artificial neural network methods
*/

#include "TMVA/TNeuron.h"

#include "TMVA/MsgLogger.h"
#include "TMVA/TActivation.h"
#include "TMVA/Tools.h"
#include "TMVA/TNeuronInput.h"
#include "TMVA/Types.h"

#include "TH1D.h"
#include "ThreadLocalStorage.h"
#include "TObjArray.h"

static const Int_t UNINITIALIZED = -1;

using std::vector;

ClassImp(TMVA::TNeuron);

////////////////////////////////////////////////////////////////////////////////
/// standard constructor

TMVA::TNeuron::TNeuron()
{
   InitNeuron();
}

////////////////////////////////////////////////////////////////////////////////
/// destructor

TMVA::TNeuron::~TNeuron()
{
   if (fLinksIn != NULL)  delete fLinksIn;
   if (fLinksOut != NULL) delete fLinksOut;
}

////////////////////////////////////////////////////////////////////////////////
/// initialize the neuron, most variables still need to be set via setters

void TMVA::TNeuron::InitNeuron()
{
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

////////////////////////////////////////////////////////////////////////////////
/// force the value, typically for input and bias neurons

void TMVA::TNeuron::ForceValue(Double_t value)
{
   fValue = value;
   fForcedValue = kTRUE;
}

////////////////////////////////////////////////////////////////////////////////
/// calculate neuron input

void TMVA::TNeuron::CalculateValue()
{
   if (fForcedValue) return;
   fValue = fInputCalculator->GetInput(this);
}

////////////////////////////////////////////////////////////////////////////////
/// calculate neuron activation/output

void TMVA::TNeuron::CalculateActivationValue()
{
   if (fActivation == NULL) {
      PrintMessage( kWARNING ,"No activation equation specified." );
      fActivationValue = UNINITIALIZED;
      return;
   }
   fActivationValue = fActivation->Eval(fValue);
}

////////////////////////////////////////////////////////////////////////////////
/// calculate error field

void TMVA::TNeuron::CalculateDelta()
{
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
      // memory for the pointer each time. Thanks to Peter Elmer who pointed this out
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

////////////////////////////////////////////////////////////////////////////////
/// set input calculator

void TMVA::TNeuron::SetInputCalculator(TNeuronInput* calculator)
{
   if (fInputCalculator != NULL) delete fInputCalculator;
   fInputCalculator = calculator;
}

////////////////////////////////////////////////////////////////////////////////
/// set activation equation

void TMVA::TNeuron::SetActivationEqn(TActivation* activation)
{
   if (fActivation != NULL) delete fActivation;
   fActivation = activation;
}

////////////////////////////////////////////////////////////////////////////////
/// add synapse as a pre-link to this neuron

void TMVA::TNeuron::AddPreLink(TSynapse* pre)
{
   if (IsInputNeuron()) return;
   fLinksIn->Add(pre);
}

////////////////////////////////////////////////////////////////////////////////
/// add synapse as a post-link to this neuron

void TMVA::TNeuron::AddPostLink(TSynapse* post)
{
   if (IsOutputNeuron()) return;
   fLinksOut->Add(post);
}

////////////////////////////////////////////////////////////////////////////////
/// delete all pre-links

void TMVA::TNeuron::DeletePreLinks()
{
   DeleteLinksArray(fLinksIn);
}

////////////////////////////////////////////////////////////////////////////////
/// delete an array of TSynapses

void TMVA::TNeuron::DeleteLinksArray(TObjArray*& links)
{
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

////////////////////////////////////////////////////////////////////////////////
/// set error, this should only be done for an output neuron

void TMVA::TNeuron::SetError(Double_t error)
{
   if (!IsOutputNeuron())
      PrintMessage( kWARNING, "Warning! Setting an error on a non-output neuron is probably not what you want to do." );

   fError = error;
}

////////////////////////////////////////////////////////////////////////////////
/// update and adjust the pre-synapses for each neuron (input neuron has no pre-synapse)
/// this method should only be called in batch mode

void TMVA::TNeuron::UpdateSynapsesBatch()
{
   if (IsInputNeuron()) return;

   TSynapse* synapse = NULL;
   TObjArrayIter iter(fLinksIn);
   while (true) {
      synapse = (TSynapse*) iter.Next();
      if (synapse == NULL) break;
      synapse->CalculateDelta();
   }

}

////////////////////////////////////////////////////////////////////////////////
/// update the pre-synapses for each neuron (input neuron has no pre-synapse)
/// this method should only be called in sequential mode

void TMVA::TNeuron::UpdateSynapsesSequential()
{
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

////////////////////////////////////////////////////////////////////////////////
/// adjust the pre-synapses' weights for each neuron (input neuron has no pre-synapse)
/// this method should only be called in batch mode

void TMVA::TNeuron::AdjustSynapseWeights()
{
   if (IsInputNeuron()) return;

   TSynapse* synapse = NULL;
   TObjArrayIter iter(fLinksIn);


   while (true) {
      synapse = (TSynapse*) iter.Next();
      if (synapse == NULL) break;
      synapse->AdjustWeight();
   }

}

////////////////////////////////////////////////////////////////////////////////
/// initialize the error fields of all pre-neurons
/// this method should only be called in batch mode

void TMVA::TNeuron::InitSynapseDeltas()
{
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

////////////////////////////////////////////////////////////////////////////////
/// print an array of TSynapses, for debugging

void TMVA::TNeuron::PrintLinks(TObjArray* links) const
{
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

////////////////////////////////////////////////////////////////////////////////
/// print activation equation, for debugging

void TMVA::TNeuron::PrintActivationEqn()
{
   if (fActivation != NULL) Log() << kDEBUG << fActivation->GetExpression() << Endl;
   else                     Log() << kDEBUG << "<none>" << Endl;
}

////////////////////////////////////////////////////////////////////////////////
/// print message, for debugging

void TMVA::TNeuron::PrintMessage( EMsgType type, TString message)
{
   Log() << type << message << Endl;
}

////////////////////////////////////////////////////////////////////////////////

TMVA::MsgLogger& TMVA::TNeuron::Log() const
{
   TTHREAD_TLS_DECL_ARG2(MsgLogger,logger,"TNeuron",kDEBUG);    //! message logger, static to save resources
   return logger;
}
