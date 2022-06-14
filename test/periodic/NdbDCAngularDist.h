#ifndef DCANGULARDIST_H
#define DCANGULARDIST_H

#include "NdbMF.h"

/* ========= NdbDCAngularDist ============ */
class NdbDCAngularDist : public NdbMF
{
protected:

public:
   NdbDCAngularDist()
      : NdbMF(34, "Data covariances for angular distributions") {}
   ~NdbDCAngularDist() {}

   ClassDef(NdbDCAngularDist,1)

}; // NdbDCAngularDist

#endif
