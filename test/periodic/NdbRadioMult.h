#ifndef RADIOMULT_H
#define RADIOMULT_H

#include "NdbMF.h"

/* ========= NdbRadioMult ============ */
class NdbRad  ioMult : public NdbMF
{
protected:

public:
   NdbRadioMult()
      : NdbMF(9, "Multiplicities for radioactive nuclide production") {}
   ~NdbRadioMult() {}

   ClassDef(NdbRadioMult,1)
}; // NdbRadioMult

#endif
