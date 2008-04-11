// @(#)root/eg:$Id$
// Author: Pasha Murat   12/02/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/
#ifndef TParticleClassPDG_hh
#define TParticleClassPDG_hh

#include "TNamed.h"
#include "TObjArray.h"

class TParticlePDG;

class TParticleClassPDG : public TNamed {
public:
//------------------------------------------------------------------------------
//     data members
//------------------------------------------------------------------------------
protected:
   TObjArray*  fListOfParticles;		// list of (non-owned) particles

   TParticleClassPDG(const TParticleClassPDG& pcp): TNamed(pcp), fListOfParticles(pcp.fListOfParticles) { }
   TParticleClassPDG& operator=(const TParticleClassPDG& pcp) 
   {if(this!=&pcp) {TNamed::operator=(pcp); fListOfParticles=pcp.fListOfParticles;}
       return *this;
   }
//------------------------------------------------------------------------------
// functions
//------------------------------------------------------------------------------
public:
   // ****** constructors  and destructor

   TParticleClassPDG(const char* name = 0);
   virtual ~TParticleClassPDG();
   // ****** access methods
  
   Int_t   GetNParticles () { 
      return fListOfParticles->GetEntriesFast();
   }

   TParticlePDG* GetParticle(Int_t i) { 
      return (TParticlePDG*) fListOfParticles->At(i); 
   }

   TObjArray* GetListOfParticles() { return fListOfParticles; }

   // ****** modifiers

   void AddParticle(TObject* p) { fListOfParticles->Add(p); }

   // ****** overloaded methods of TObject

   virtual void    Print(Option_t* opt="") const; // *MENU*

   Bool_t IsFolder() const { return kTRUE; }
   virtual void   Browse(TBrowser* b);

   ClassDef(TParticleClassPDG,1)		// PDG static particle definition
};

#endif
