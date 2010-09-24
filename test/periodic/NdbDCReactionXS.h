#ifndef DCREACTIONXS_H
#define DCREACTIONXS_H

#include "NdbMF.h"

/* ========= NdbDCReactionXS ============ */
class NdbDCReactionXS : public NdbMF
{
protected:

public:
	NdbDCReactionXS()
		: NdbMF(33, "Data covariances for reaction cross section") {}

	~NdbDCReactionXS() {}

	ClassDef(NdbDCReactionXS,1)

}; // NdbDCReactionXS

#endif
