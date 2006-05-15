// @(#)root/eg:$Name:  $:$Id: TParticleClassPDG.cxx,v 1.3 2005/09/04 11:42:05 brun Exp $
// Author: Pasha Murat   12/02/99

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


