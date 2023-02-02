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

   ~NdbGeneralInfo() override {}

   ClassDefOverride(NdbGeneralInfo,1)

}; // NdbGeneralInfo

#endif
