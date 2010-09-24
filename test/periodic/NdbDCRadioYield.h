#ifndef DCRADIOYIELD_H
#define DCRADIOYIELD_H

#include "NdbMF.h"

/* ========= NdbDCRadioYield ============ */
class NdbDCRadioYield : public NdbMF
{
protected:

public:
	NdbDCRadioYield()
		: NdbMF(39, "Data covariances for radionuclide "
			"production yields") {}
	~NdbDCRadioYield() {}

	ClassDef(NdbDCRadioYield,1)

}; // NdbDCRadioYield

#endif
