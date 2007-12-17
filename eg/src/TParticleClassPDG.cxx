// @(#)root/eg:$Id$
// Author: Pasha Murat   12/02/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

//______________________________________________________________________________
//
// Utility class used internally by TDatabasePDG

#include "TDecayChannel.h"
#include "TParticlePDG.h"
#include "TParticleClassPDG.h"

ClassImp(TParticleClassPDG)

//______________________________________________________________________________
TParticleClassPDG::TParticleClassPDG(const char* name): TNamed(name,name)
{
   //default constructor
   fListOfParticles  = new TObjArray(5);
}

//______________________________________________________________________________
TParticleClassPDG::~TParticleClassPDG() {
   // destructor, class doesn't own its particles...

   delete fListOfParticles;
}



//______________________________________________________________________________
void TParticleClassPDG::Print(Option_t *) const
{
//
//  Print the entire information of this kind of particle
//

   printf("Particle class: %-20s",GetName());
   if (fListOfParticles) {
      int banner_printed = 0;
      TIter next(fListOfParticles);
      TParticlePDG *p;
      while ((p = (TParticlePDG*)next())) {
         if (! banner_printed) {
            p->Print("banner");
            banner_printed = 1;
         }
         p->Print("");
      }
   }
}

//______________________________________________________________________________
void TParticleClassPDG::Browse(TBrowser* b)
{
   //browse this particle class
   if (fListOfParticles) fListOfParticles->Browse(b);
}


