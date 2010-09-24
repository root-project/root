#ifndef NDBMF_H
#define NDBMF_H

#include <TObject.h>
#include <TString.h>
#include <TObjArray.h>

#include "NdbMT.h"

/* ================== NdbMF =================== */
/*
 * Base class for the File Types (MF)
 */
class NdbMF: public TObject
{
protected:
	Int_t		iMF;		// MF number
	TString		sDescription;	// Description of the File TYPE (MF)
	TObjArray	aSection;	// Array of sections (MT)

public:
	NdbMF( Int_t aMF, const char *aDesc)
	: TObject(), sDescription(aDesc), aSection() {
		iMF = aMF;
	}

	~NdbMF() {}

	// Virtual functions
	virtual Int_t Compare(const TObject *o) const
		{ return ((iMF==((NdbMF*)o)->iMF)? 0 :
				(iMF > ((NdbMF*)o)->iMF)? 1 : -1 ); }

	// Access functions
	inline Int_t MF()		const { return iMF; }
	inline TString Description()	const { return sDescription; }

	// Find Section, by type
	NdbMT*	Section(Int_t id);

	// Find Section by name
	NdbMT*	Section(const char *name);

	// Add a new section in the section list
	void	Add(NdbMT& sec);

	// Enumerate sections in ENDF file
	Int_t	EnumerateENDFType( Int_t sec );

	// Move File pointer to beggining of MF section in ENDF file
	void	LocateType() {}

	ClassDef(NdbMF,1)

}; // NdbMF

#endif
