// @(#)root/eg:$Name$:$Id$
// Author: Pasha Murat   12/02/99

#include "TParticlePDG.h"


ClassImp(TParticlePDG)

//______________________________________________________________________________
TParticlePDG::TParticlePDG()
{
  fDecayList  = NULL;
}

//______________________________________________________________________________
TParticlePDG::TParticlePDG(Int_t )
{
  // empty for the time  being

  fDecayList  = NULL;
}

//______________________________________________________________________________
TParticlePDG::TParticlePDG(const char* name, const char* title, Double_t mass,
			   Bool_t stable, Double_t width, Double_t charge,
			   const char* type, Int_t MCnumber)
             : TNamed(name,title)
{

    // empty for the time  being

    fMass       = mass;
    fStable     = stable;
    fWidth      = width;
    fCharge     = charge;
    fType       = type;
    fPdgCode    = MCnumber;
    fDecayList  = NULL;
}


//______________________________________________________________________________
TParticlePDG::~TParticlePDG()
{
}

//______________________________________________________________________________
void TParticlePDG::Print(Option_t *)
{
//
//  Print the entire information of this kind of particle
//

   printf("\n%-20s  %6d\t",GetName(),fPdgCode);
   if (!fStable) {
       printf("Mass:%9.4f Width (GeV):%11.4e\tCharge: %5.1f\n",
              fMass, fWidth, fCharge);
   }
   else {
       printf("Mass:%9.4f Width (GeV): Stable\tCharge: %5.1f\n",
              fMass, fCharge);
   }
}

