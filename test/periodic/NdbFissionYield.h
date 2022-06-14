#ifndef FISSIONYIELD_H
#define FISSIONYIELD_H

#include "NdbMF.h"

/* ========= NdbFissionYield ============ */
class NdbFissionYield : public NdbMF
{
protected:

public:
   NdbFissionYield()
      : NdbMF(8, "Radioactivity and fission-product yield data") {}
   ~NdbFissionYield() {}

   ClassDef(NdbFissionYield,1)

}; // NdbFissionYield

#endif
