// @(#)root/mlp:$Name:  $:$Id: TSynapse.cxx,v 1.1 2003/08/27 15:31:14 brun Exp $
// Author: Christophe.Delaere@cern.ch   21/08/2002

///////////////////////////////////////////////////////////////////////////
//
// TSynapse
//
// This is a simple weighted bidirectionnal connection between 
// two neurons.
// A network is built connecting two neurons by a synapse.
// In addition to the value, the synapse can return the DeDw
//
///////////////////////////////////////////////////////////////////////////

#include "TSynapse.h"
#include "TNeuron.h"
#include "Riostream.h"

ClassImp(TSynapse)

//______________________________________________________________________________
TSynapse::TSynapse()
{
   // Default constructor
   fpre = NULL;
   fpost = NULL;
   fweight = 1;
}

//______________________________________________________________________________
TSynapse::TSynapse(TNeuron * pre, TNeuron * post, Double_t w)
{
   // Constructor that connects two neurons
   fpre = pre;
   fpost = post;
   fweight = w;
   pre->AddPost(this);
   post->AddPre(this);
}

//______________________________________________________________________________
void TSynapse::SetPre(TNeuron * pre)
{
   // Sets the pre-neuron
   if (pre) {
      Error("SetPre","this synapse is already assigned to a pre-neuron.");
      return;
   }
   fpre = pre;
   pre->AddPost(this);
}

//______________________________________________________________________________
void TSynapse::SetPost(TNeuron * post)
{
   // Sets the post-neuron
   if (post) {
      Error("SetPost","this synapse is already assigned to a post-neuron.");
      return;
   }
   fpost = post;
   post->AddPre(this);
}

//______________________________________________________________________________
Double_t TSynapse::GetValue()
{
   // Returns the value: weithted input
   if (fpre)
      return (fweight * fpre->GetValue());
   return 0;
}

//______________________________________________________________________________
Double_t TSynapse::GetDeDw()
{
   // Computes the derivative of the error wrt the synapse weight.
   if (!(fpre && fpost))
      return 0;
   return (fpre->GetValue() * fpost->GetDeDw());
}

//______________________________________________________________________________
void TSynapse::SetWeight(Double_t w) 
{ 
   // Sets the weight of the synapse.
   // This weight is the multiplying factor applied on the 
   // output of a neuron in the linear combination given as input 
   // of another neuron.
   fweight = w; 
}

//______________________________________________________________________________
void TSynapse::SetDEDw(Double_t in) 
{ 
   // Sets the derivative of the total error wrt the synapse weight
   fDEDw = in; 
}


