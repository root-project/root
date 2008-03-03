// @(#)root/mlp:$Id$
// Author: Christophe.Delaere@cern.ch   20/07/03

/*************************************************************************
 * Copyright (C) 1995-2003, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TNeuron
#define ROOT_TNeuron

#ifndef ROOT_TNamed
#include "TNamed.h"
#endif
#ifndef ROOT_TObjArray
#include "TObjArray.h"
#endif

class TTreeFormula;
class TSynapse;
class TBranch;
class TTree;
class TFormula;

//____________________________________________________________________
//
// TNeuron
//
// This class decribes an elementary neuron, which is the basic
// element for a Neural Network.
// A network is build connecting neurons by synapses.
// There are different types of neurons: linear (a+bx),
// sigmoid (1/(1+exp(-x)), tanh or gaussian.
// In a Multi Layer Perceptron, the input layer is made of
// inactive neurons (returning the normalized input), hidden layers
// are made of sigmoids and output neurons are linear.
//
// This implementation contains several methods to compute the value,
// the derivative, the DeDw, ...
// Values are stored in local buffers. The SetNewEvent() method is
// there to inform buffered values are outdated.
//
//____________________________________________________________________

class TNeuron : public TNamed {
   friend class TSynapse;

 public:
   enum ENeuronType { kOff, kLinear, kSigmoid, kTanh, kGauss, kSoftmax, kExternal };

   TNeuron(ENeuronType type = kSigmoid, 
           const char* name = "", const char* title = "", 
           const char* extF = "", const char* extD  = "" );
   virtual ~TNeuron() {}
   inline TSynapse* GetPre(Int_t n) const { return (TSynapse*) fpre.At(n); }
   inline TSynapse* GetPost(Int_t n) const { return (TSynapse*) fpost.At(n); }
   inline TNeuron* GetInLayer(Int_t n) const { return (TNeuron*) flayer.At(n); }
   TTreeFormula* UseBranch(TTree*, const char*);
   Double_t GetInput() const;
   Double_t GetValue() const;
   Double_t GetDerivative() const;
   Double_t GetError() const;
   Double_t GetTarget() const;
   Double_t GetDeDw() const;
   Double_t GetBranch() const;
   ENeuronType GetType() const;
   void SetWeight(Double_t w);
   inline Double_t GetWeight() const { return fWeight; }
   void SetNormalisation(Double_t mean, Double_t RMS);
   inline const Double_t* GetNormalisation() const { return fNorm; }
   void SetNewEvent() const;
   void SetDEDw(Double_t in);
   inline Double_t GetDEDw() const { return fDEDw; }
   void ForceExternalValue(Double_t value);
   void AddInLayer(TNeuron*);

 protected:
   Double_t Sigmoid(Double_t x) const;
   Double_t DSigmoid(Double_t x) const;
   void AddPre(TSynapse*);
   void AddPost(TSynapse*);

 private:
   TNeuron(const TNeuron&); // Not implemented
   TNeuron& operator=(const TNeuron&); // Not implemented

   TObjArray fpre;        // pointers to the previous level in a network
   TObjArray fpost;       // pointers to the next level in a network
   TObjArray flayer;      // pointers to the current level in a network (neurons, not synapses)
   Double_t fWeight;      // weight used for computation
   Double_t fNorm[2];     // normalisation to mean=0, RMS=1.
   ENeuronType fType;     // neuron type
   TFormula* fExtF;       // function   (external mode)
   TFormula* fExtD;       // derivative (external mode)
   //buffers
   //should be mutable when supported by all compilers
   TTreeFormula* fFormula;//! formula to be used for inputs and outputs
   Int_t fIndex;          //! index in the formula
   Bool_t fNewInput;      //! do we need to compute fInput again ?
   Double_t fInput;       //! buffer containing the last neuron input
   Bool_t fNewValue;      //! do we need to compute fValue again ?
   Double_t fValue;       //! buffer containing the last neuron output
   Bool_t fNewDeriv;      //! do we need to compute fDerivative again ?
   Double_t fDerivative;  //! buffer containing the last neuron derivative
   Bool_t fNewDeDw;       //! do we need to compute fDeDw again ?
   Double_t fDeDw;        //! buffer containing the last derivative of the error
   Double_t fDEDw;        //! buffer containing the sum over all examples of DeDw

   ClassDef(TNeuron, 4)   // Neuron for MultiLayerPerceptrons
};

#endif
