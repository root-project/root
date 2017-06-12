#include <stdlib.h>
#include <Riostream.h>
#include "NdbParticleList.h"

ClassImp(NdbParticleList);

/* -------- TotalCharge -------- */
Int_t
NdbParticleList::TotalCharge()
{
   Int_t   charge = 0;

   for (int i=0; i<mult.GetSize(); i++)
      charge += ((NdbParticle*)part[i])->Charge() * mult[i];

   return charge;
} // TotalCharge

/* -------- TotalMass -------- */
Float_t
NdbParticleList::TotalMass()
{
   Float_t   mass = 0.0;

   for (int i=0; i<mult.GetSize(); i++)
      mass += ((NdbParticle*)part[i])->Mass() * mult[i];

   return mass;
} // TotalMass

/* -------- Name -------- */
TString
NdbParticleList::Name()
{
   TString   nm;

   for (int i=0; i<mult.GetSize(); i++) {
      if (mult[i]>1) {
         char   num[10];
         sprintf(num,"%d",mult[i]);
         nm.Append(num);
      }
      nm.Append(((NdbParticle*)part[i])->Name());
   }
   return nm;
} // Name

/* --------- Add --------- */
void
NdbParticleList::Add(NdbParticle *, Int_t)
{
   std::cout << "NdbParticleList::add()" << std::endl;
} // Add
