// @(#)root/eg:$Id$
// Author: Pasha Murat   12/02/99

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

/** \class  TParticleClassPDG
    \ingroup eg

Utility class used internally by TDatabasePDG
*/

#include "TDecayChannel.h"
#include "TParticlePDG.h"
#include "TParticleClassPDG.h"

ClassImp(TParticleClassPDG);

////////////////////////////////////////////////////////////////////////////////
///default constructor

TParticleClassPDG::TParticleClassPDG(const char* name): TNamed(name,name)
{
   fListOfParticles  = new TObjArray(5);
}

////////////////////////////////////////////////////////////////////////////////
/// destructor, class doesn't own its particles...

TParticleClassPDG::~TParticleClassPDG() {
   delete fListOfParticles;
}



////////////////////////////////////////////////////////////////////////////////
///
///  Print the entire information of this kind of particle
///

void TParticleClassPDG::Print(Option_t *) const
{
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

////////////////////////////////////////////////////////////////////////////////
///browse this particle class

void TParticleClassPDG::Browse(TBrowser* b)
{
   if (fListOfParticles) fListOfParticles->Browse(b);
}


