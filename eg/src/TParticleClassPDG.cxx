// @(#)root/eg:$Name:  $:$Id: TParticleClassPDG.cc,v 1.1 2001/02/22 14:17:20 murat Exp $
// Author: Pasha Murat   12/02/99

#include "TDecayChannel.h"
#include "TParticlePDG.h"
#include "TParticleClassPDG.h"

ClassImp(TParticleClassPDG)

//______________________________________________________________________________
TParticleClassPDG::TParticleClassPDG(const char* name): TNamed(name,name)
{
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
    TIter it(fListOfParticles);
    while (TParticlePDG* p = (TParticlePDG*) it.Next()) {
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
  if (fListOfParticles) fListOfParticles->Browse(b);
}


