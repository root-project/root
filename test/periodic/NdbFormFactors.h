#ifndef MFFORMFACTORS_H
#define MFFORMFACTORS_H

#include "NdbMF.h"

/* ========= NdbFormFactors ============ */
class NdbFormFactors : public NdbMF
{
protected:

public:
	NdbFormFactors()
		: NdbMF(27, "Atomic form factors or scattering functions "
			"for photo-atomic interactions") {}
	~NdbFormFactors() {}

	ClassDef(NdbFormFactors,1)

}; // NdbFormFactors

#endif
