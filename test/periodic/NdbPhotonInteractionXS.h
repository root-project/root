#ifndef PHOTONINTERACTIONXS_H
#define PHOTONINTERACTIONXS_H

#include "NdbMF.h"

/* ========= NdbPhotonInteractionXS ============ */
class NdbPhotonInteractionXS : public NdbMF
{
protected:

public:
   NdbPhotonInteractionXS()
      : NdbMF(23, "Photo-atomic interaction cross sections") {}
   ~NdbPhotonInteractionXS() override {}

   ClassDefOverride(NdbPhotonInteractionXS,1)
}; // NdbPhotonInteractionXS

#endif
