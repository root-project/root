// @(#)root/mlp:$Id$
// Author: Christophe.Delaere@cern.ch   20/07/2003

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TSynapse
#define ROOT_TSynapse

#ifndef ROOT_TObject
#include "TObject.h"
#endif

class TNeuron;

//____________________________________________________________________
//
// TSynapse
//
// This is a simple weighted bidirectionnal connection between
// two neurons.
// A network is built connecting two neurons by a synapse.
// In addition to the value, the synapse can return the DeDw
//
//____________________________________________________________________

class TSynapse : public TObject {
 public:
   TSynapse();
   TSynapse(TNeuron*, TNeuron*, Double_t w = 1);
   virtual ~TSynapse() {}
   void SetPre(TNeuron* pre);
   void SetPost(TNeuron* post);
   inline TNeuron* GetPre()  const { return fpre; }
   inline TNeuron* GetPost() const { return fpost; }
   void SetWeight(Double_t w);
   inline Double_t GetWeight() const { return fweight; }
   Double_t GetValue() const;
   Double_t GetDeDw() const;
   void SetDEDw(Double_t in);
   Double_t GetDEDw() const { return fDEDw; }

 private:
   TNeuron* fpre;         // the neuron before the synapse
   TNeuron* fpost;        // the neuron after the synapse
   Double_t fweight;      // the weight of the synapse
   Double_t fDEDw;        //! the derivative of the total error wrt the synapse weight

   ClassDef(TSynapse, 1)  // simple weighted bidirectionnal connection between 2 neurons
};

#endif
