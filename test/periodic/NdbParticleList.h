#ifndef PARTICLELIST_H
#define PARTICLELIST_H

#include <TArrayI.h>
#include <TObjArray.h>

#include "NdbParticle.h"

/* ------------ Particle List ---------------- */
class NdbParticleList : public TObject
{
protected:
	TArrayI		mult;		// How many times each particle occurs
	TObjArray	part;		// Link to particle types

public:
	NdbParticleList() : mult(), part() { }
	~NdbParticleList() {}

	// --- Access Functions ---
	Int_t	TotalCharge();		// Total charge
	Float_t	TotalMass();		// Total mass
	TString	Name();			// string containing the name
					// in a format like 2np

	void	Add(NdbParticle *p, Int_t n=1);

	ClassDef(NdbParticleList,1)

}; // NdbParticleList

#endif
