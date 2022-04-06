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

#include "TObject.h"

class TNeuron;


class TSynapse : public TObject {
 public:
   TSynapse();
   TSynapse(TNeuron*, TNeuron*, Double_t w = 1);
   ~TSynapse() override {}
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
   TNeuron* fpre;         ///< the neuron before the synapse
   TNeuron* fpost;        ///< the neuron after the synapse
   Double_t fweight;      ///< the weight of the synapse
   Double_t fDEDw;        ///<! the derivative of the total error wrt the synapse weight

   ClassDefOverride(TSynapse, 1)  ///< simple weighted bidirectional connection between 2 neurons
};

#endif
