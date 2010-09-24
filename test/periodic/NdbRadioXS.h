#ifndef RADIOXS_H
#define RADIOXS_H

#include "NdbMF.h"

/* ========= NdbRadioXS ============ */
class NdbRadioXS : public NdbMF
{
protected:

public:
	NdbRadioXS()
		: NdbMF(10, "Cross section for radioactive nuclide production") {}

	~NdbRadioXS() {}

	ClassDef(NdbRadioXS,1)
}; // NdbRadioXS

#endif
