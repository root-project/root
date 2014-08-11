#ifndef PHOTONMULT_H
#define PHOTONMULT_H

#include "NdbMF.h"

/* ========= NdbPhotonMult ============ */
class NdbPhotonMult : public NdbMF
{
protected:

public:
   NdbPhotonMult()
      : NdbMF(12, "Multiplicities for photon production") {}
   ~NdbPhotonMult() {}

   ClassDef(NdbPhotonMult,1)
}; // NdbPhotonMult

#endif
