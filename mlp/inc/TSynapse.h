// @(#)root/mlp:$Name:  $:$Id: TSynapse.cxx,v 1.00 2003/08/27 13:52:36 brun Exp $
// Author: Christophe.Delaere@cern.ch   20/07/2003

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
   virtual ~ TSynapse() {} 
   void SetPre(TNeuron* pre);
   void SetPost(TNeuron* post);
   inline TNeuron* GetPre()  { return fpre; }
   inline TNeuron* GetPost() { return fpost; }
   void SetWeight(Double_t w); 
   inline Double_t GetWeight() { return fweight; }
   Double_t GetValue();
   Double_t GetDeDw();
   void SetDEDw(Double_t in); 
   Double_t GetDEDw() { return fDEDw; }
   
 private:
   TNeuron* fpre;         // the neuron before the synapse
   TNeuron* fpost;        // the neuron after the synapse
   Double_t fweight;      // the weight of the synapse
   Double_t fDEDw;        //! the derivative of the total error wrt the synapse weight
   
   ClassDef(TSynapse, 1)  // simple weighted bidirectionnal connection between 2 neurons
};

#endif
