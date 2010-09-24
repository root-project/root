#ifndef REACTION_H
#define REACTION_H

#include <TObjArray.h>

#include "NdbMF.h"
#include "NdbMTReactionXS.h"
#include "NdbParticleList.h"

/* ========= NdbReaction ============ */
class NdbReaction : public NdbMF
{
protected:
	TObjArray	reac;		// List of reaction Cross Sections

public:
	NdbReaction()
		: NdbMF(3, "Reaction cross sections"),
		  reac() { }
	~NdbReaction() {}

	ClassDef(NdbReaction,1)
}; // NdbReaction

#endif
