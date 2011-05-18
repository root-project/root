#ifndef THERMALNEUTRON_H
#define THERMALNEUTRON_H

#include "NdbMF.h"

/* ========= NdbThermalNeutron ============ */
class NdbThermalNeutron : public NdbMF
{
protected:

public:
	NdbThermalNeutron()
		: NdbMF(7, "Thermal neutron scattering law data") {}

	~NdbThermalNeutron() {}

	ClassDef(NdbThermalNeutron,1)
}; // NdbThermalNeutron

#endif
