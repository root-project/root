#ifndef ENERGYDIST_H
#define ENERGYDIST_H

#include "NdbMF.h"

/* ========= NdbEnergyDist ============ */
class NdbEnergyDist : public NdbMF
{
protected:

public:
   NdbEnergyDist()
      : NdbMF(5, "Energy distributions for emitted particles") {}

   ~NdbEnergyDist() {}

   ClassDef(NdbEnergyDist,1)

}; // NdbEnergyDist

#endif
