#ifndef ENERGYANGLEDIST_H
#define ENERGYANGLEDIST_H

#include "NdbMF.h"

/* ========= NdbEnergyAngleDist ============ */
class NdbEnergyAngleDist : public NdbMF
{
protected:

public:
	NdbEnergyAngleDist()
		: NdbMF(6, "Energy-angle distributions for emitted particles") {}

	~NdbEnergyAngleDist() {}

	ClassDef(NdbEnergyAngleDist,1)

}; // NdbEnergyAngleDist

#endif
