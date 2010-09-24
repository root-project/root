#ifndef DCRESPARAM_H
#define DCRESPARAM_H

#include "NdbMF.h"

/* ========= NdbDCResParam ============ */
class NdbDCResParam : public NdbMF
{
protected:

public:
	NdbDCResParam()
		: NdbMF(32, "Data covariances for resonance parameters") {}
	~NdbDCResParam() {}

	ClassDef(NdbDCResParam,1)

}; // NdbDCResParam

#endif
