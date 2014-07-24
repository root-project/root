#ifndef PHOTONANGLEDIST_H
#define PHOTONANGLEDIST_H

#include "NdbMF.h"

/* ========= NdbPhotonAngleDist ============ */
class NdbPhotonAngleDist : public NdbMF
{
protected:

public:
   NdbPhotonAngleDist()
      : NdbMF(14, "Angular distributions for photon production") {}

   ~NdbPhotonAngleDist() {}

   ClassDef(NdbPhotonAngleDist,1)
}; // NdbPhotonAngleDist

#endif
