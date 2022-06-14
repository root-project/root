// @(#)root/tmva $Id$
// Author: Matt Jachowski

/**********************************************************************************
 * Project: TMVA - a Root-integrated toolkit for multivariate data analysis       *
 * Package: TMVA                                                                  *
 * Class  : TMVA::TNeuron                                                         *
 * Web    : http://tmva.sourceforge.net                                           *
 *                                                                                *
 * Description:                                                                   *
 *      Neuron class to be used in MethodANNBase and its derivatives.             *
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

#ifndef ROOT_TMVA_TNeuron
#define ROOT_TMVA_TNeuron

//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TNeuron                                                              //
//                                                                      //
// Neuron used by derivatives of MethodANNBase                          //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#include <iostream>

#include "TString.h"
#include "TObjArray.h"
#include "TFormula.h"

#include "TMVA/TSynapse.h"
#include "TMVA/TActivation.h"
#include "TMVA/Types.h"

namespace TMVA {

   class TNeuronInput;

   class TNeuron : public TObject {

   public:

      TNeuron();
      virtual ~TNeuron();

      // force the input value
      void ForceValue(Double_t value);

      // calculate the input value
      void CalculateValue();

      // calculate the activation value
      void CalculateActivationValue();

      // calculate the error field of the neuron
      void CalculateDelta();

      // set the activation function
      void SetActivationEqn(TActivation* activation);

      // set the input calculator
      void SetInputCalculator(TNeuronInput* calculator);

      // add a synapse as a pre-link
      void AddPreLink(TSynapse* pre);

      // add a synapse as a post-link
      void AddPostLink(TSynapse* post);

      // delete all pre-links
      void DeletePreLinks();

      // set the error
      void SetError(Double_t error);

      // update the error fields of all pre-synapses, batch mode
      // to actually update the weights, call adjust synapse weights
      void UpdateSynapsesBatch();

      // update the error fields and weights of all pre-synapses, sequential mode
      void UpdateSynapsesSequential();

      // update the weights of the all pre-synapses, batch mode
      //(call UpdateSynapsesBatch first)
      void AdjustSynapseWeights();

      // explicitly initialize error fields of pre-synapses, batch mode
      void InitSynapseDeltas();

      // print activation equation, for debugging
      void PrintActivationEqn();

      // inlined functions
      Double_t  GetValue() const                { return fValue;                          }
      Double_t  GetActivationValue() const      { return fActivationValue;                }
      Double_t  GetDelta() const                { return fDelta;                          }
      Double_t  GetDEDw() const                 { return fDEDw;                           }
      Int_t     NumPreLinks() const             { return NumLinks(fLinksIn);              }
      Int_t     NumPostLinks() const            { return NumLinks(fLinksOut);             }
      TSynapse* PreLinkAt ( Int_t index ) const { return (TSynapse*)fLinksIn->At(index);  }
      TSynapse* PostLinkAt( Int_t index ) const { return (TSynapse*)fLinksOut->At(index); }
      void      SetInputNeuron()                { NullifyLinks(fLinksIn);                 }
      void      SetOutputNeuron()               { NullifyLinks(fLinksOut);                }
      void      SetBiasNeuron()                 { NullifyLinks(fLinksIn);                 }
      void      SetDEDw( Double_t DEDw )        { fDEDw = DEDw;                           }
      Bool_t    IsInputNeuron() const           { return fLinksIn == NULL;                }
      Bool_t    IsOutputNeuron() const          { return fLinksOut == NULL;               }
      void      PrintPreLinks() const           { PrintLinks(fLinksIn); return;           }
      void      PrintPostLinks() const          { PrintLinks(fLinksOut); return;          }

      virtual void Print(Option_t* = "") const {
         std::cout << fValue << std::endl;
         //PrintPreLinks(); PrintPostLinks();
      }

   private:

      // private helper functions
      void InitNeuron();
      void DeleteLinksArray( TObjArray*& links );
      void PrintLinks      ( TObjArray* links ) const;
      void PrintMessage    ( EMsgType, TString message );

      // inlined helper functions
      Int_t NumLinks(TObjArray* links) const {
         if (links == nullptr) return 0;
         else return links->GetEntriesFast();
      }
      void NullifyLinks(TObjArray*& links) {
         if (links != nullptr) { delete links; links = nullptr; }
      }

      // private member variables
      TObjArray*    fLinksIn;                 ///< array of input synapses
      TObjArray*    fLinksOut;                ///< array of output synapses
      Double_t      fValue;                   ///< input value
      Double_t      fActivationValue;         ///< activation/output value
      Double_t      fDelta;                   ///< error field of neuron
      Double_t      fDEDw;                    ///< sum of all deltas
      Double_t      fError;                   ///< error, only set for output neurons
      Bool_t        fForcedValue;             ///< flag for forced input value
      TActivation*  fActivation;              ///< activation equation
      TNeuronInput* fInputCalculator;         ///< input calculator

      MsgLogger& Log() const;

      ClassDef(TNeuron,0); // Neuron class used by MethodANNBase derivative ANNs
   };

} // namespace TMVA

#endif
