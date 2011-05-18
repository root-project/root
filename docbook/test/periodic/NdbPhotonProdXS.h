#ifndef PHOTONPRODXS_H
#define PHOTONPRODXS_H

#include "NdbMF.h"

/* ========= NdbPhotonProdXS ============ */
class NdbPhotonProdXS : public NdbMF
{
protected:

public:
	NdbPhotonProdXS()
		: NdbMF(13, "Cross sections for photon production") {}

	~NdbPhotonProdXS() {}

	ClassDef(NdbPhotonProdXS,1)
}; // NdbPhotonProdXS

#endif
