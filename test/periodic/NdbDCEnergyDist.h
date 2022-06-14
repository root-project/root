#ifndef MFDCENERGYDIST_H
#define MFDCENERGYDIST_H

#include "NdbMF.h"

/* ========= NdbDCEnergyDist ============ */
class NdbDCEnergyDist : public NdbMF
{
protected:

public:
   NdbDCEnergyDist()
      : NdbMF(35, "Data covariances for energy distributions") {}
   ~NdbDCEnergyDist() {}

   ClassDef(NdbDCEnergyDist,1)

}; // NdbDCEnergyDist

#endif
