#ifndef NDBANGULARDIST_H
#define NDBANGULARDIST_H

#include "NdbMF.h"

/* ========= TMFAngularDist ============ */
class NdbAngularDist : public NdbMF
{
protected:

public:
   NdbAngularDist()
      : NdbMF(4, "Angular distributions for emitted particles") {}
   ~NdbAngularDist() {}

   ClassDef(NdbAngularDist,1)

}; // NdbMfAngularDist

#endif
