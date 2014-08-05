#ifndef MFDCRADIOXS_H
#define MFDCRADIOXS_H

#include "NdbMF.h"

/* ========= NdbDCRadioXS ============ */
class NdbDCRadioXS : public NdbMF
{
protected:

public:
   NdbDCRadioXS()
      : NdbMF(40, "Data covariances for radionuclide production "
         "cross sections") {}
   ~NdbDCRadioXS() {}

   ClassDef(NdbDCRadioXS,1)

}; // NdbDCRadioXS

#endif
