#ifndef PHOTONENERGYDIST_H
#define PHOTONENERGYDIST_H

#include "NdbMF.h"

/* ========= NdbPhotonEnergyDist ============ */
class NdbPhotonEnergyDist : public NdbMF
{
protected:

public:
   NdbPhotonEnergyDist()
      : NdbMF(15, "Energy distributions for photon production") {}
   ~NdbPhotonEnergyDist() {}

   ClassDef(NdbPhotonEnergyDist,1)
}; // NdbPhotonEnergyDist

#endif
