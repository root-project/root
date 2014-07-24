#ifndef MFDCNUBAR_H
#define MFDCNUBAR_H

#include "NdbMF.h"

/* ========= TMFDCNuBar ============ */
class NdbDCNuBar : public NdbMF
{
protected:

public:
   NdbDCNuBar()
      : NdbMF(31, "Data covariances for nu(bar)") {}
   ~NdbDCNuBar() {}

   ClassDef(NdbDCNuBar,1)

}; // NdbDCNuBar

#endif
