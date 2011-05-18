#ifndef DATACOVARIANCES_H
#define DATACOVARIANCES_H

#include "NdbMF.h"

/* ========= NdbDataCovariances ============ */
class NdbDataCovariances : public NdbMF
{
protected:

public:
	NdbDataCovariances()
		: NdbMF(30, "Data covariances obtained from parameter "
			"covariances and sensitivities") {}

	~NdbDataCovariances() {}

	ClassDef(NdbDataCovariances,1)

}; // NdbDataCovariances

#endif
