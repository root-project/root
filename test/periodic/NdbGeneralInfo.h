#ifndef GENERALINFO_H
#define GENERALINFO_H

#include "NdbMF.h"

/* ========= NdbGeneralInfo ============ */
class NdbGeneralInfo : public NdbMF
{
protected:

public:
   NdbGeneralInfo()
      : NdbMF(1, "General Information") {}

   ~NdbGeneralInfo() {}

   ClassDef(NdbGeneralInfo,1)

}; // NdbGeneralInfo

#endif
