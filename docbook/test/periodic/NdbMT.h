#ifndef MT_H
#define MT_H

#ifndef ROOT_TObject
#       include <TObject.h>
#endif

#ifndef ROOT_TString
#       include <TString.h>
#endif

#include <stdio.h>

/* ================== NdbMT =================== */
/*
 * Defines the basic object for Sections of a File Type
 */
class NdbMT: public TObject
{
protected:
	Int_t		iMT;		// MT number
	TString		sDescription;

public:
	NdbMT( Int_t aMT, const char *desc)
	: sDescription(desc) { iMT = aMT; }

	~NdbMT() {}

	// Virtual functions
	virtual Int_t Compare(const TObject *o) const
		{ return ((iMT == ((NdbMT*)o)->iMT)? 0 :
				(iMT > ((NdbMT*)o)->iMT)? 1 : -1 ); }

	// Access functions
	inline Int_t	MT()		const { return iMT; }
	inline TString	Description()	const { return sDescription; }

	// Assign a working file

	// Enumerate sections in ENDF file
	Int_t	EnumerateENDFSection( Int_t /* sec */) {return 0;}

	// Move File pointer to beggining of MT section in ENDF file
	void	LocateENDFSection() {}

	// END of ENDF Section reached?
	Bool_t	ENDF_EOS() {return 0;}

	// Abstract functions (or Virtual?)
	virtual void ReadENDFSectionHeader() {}
	virtual void ReadENDFSection() {}

	ClassDef(NdbMT,1)

}; // NdbMT

#endif
