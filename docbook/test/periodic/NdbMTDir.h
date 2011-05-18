#ifndef MTDIRXS_H
#define MTDIRXS_H

#include <TArrayI.h>

#include "NdbDefs.h"
#include "NdbMT.h"
#include "NdbMaterial.h"

/* ========= NdbMTDir ============ */
class NdbMTDir : public NdbMT
{
protected:
	Float_t		ZA;
	Float_t		AWR;
	Int_t		LRP;		// Flag for resonances in MF=2
	Int_t		LFI;		// Whether this material fissions
	Int_t		NLIB;		// Library
	Int_t		NVER;		// Version
	Int_t		NFOR;		// Format
	Float_t		ELIS;		// Excitation energy
	Int_t		STA;		// Stability flag
	Int_t		LIS;		// Ground state
	Int_t		LISO;		// Isomeric state number
	Int_t		NMOD;		// Modification number
	Float_t		AWI;		// Mass in neutron units
	Int_t		NSUB;		// Sublibrary number
	Float_t		TEMP;		// Target temperature
	Int_t		LDRV;		// Special Derived material flag
	Int_t		NWD;		// Number of records
	Int_t		NXC;		// Number of records in the dir.
	char		*ZSYMAM;	//! Symbol
	char		*ALAB;		//! Laboratory mnemonic
	char		*EDATE;		//! Evaluation date
	char		*AUTH;		//! Authors name
	char		*REF;		//! Primary reference
	char		*DDATE;		//! Distribution date
	char		*RDATE;		//! Date&Number of last revision
	char		*ENDATE;	//! Master entry date yymmdd
	TString		INFO;


	TArrayI		dir_mf;		// Directory entries
	TArrayI		dir_mt;
	TArrayI		dir_mc;
	TArrayI		dir_mod;

public:
	NdbMTDir()
	: NdbMT(451,"Descriptive Data and Directory")
	{
		NXC	= 0;
		ZSYMAM	= NULL;
		ALAB	= NULL;
		EDATE	= NULL;
		AUTH	= NULL;
		REF	= NULL;
		DDATE	= NULL;
		RDATE	= NULL;
		ENDATE	= NULL;
	}

	~NdbMTDir();

	// --- Input/Output routines ---
	Bool_t		LoadENDF(char *filename);

	inline	Int_t	Sections()	const { return NXC; }
	inline	Int_t	DIRMF(Int_t i)	{ return dir_mf[i]; }
	inline	Int_t	DIRMT(Int_t i)	{ return dir_mt[i]; }
	inline	Int_t	DIRMC(Int_t i)	{ return dir_mc[i]; }
	inline	Int_t	DIRMOD(Int_t i)	{ return dir_mod[i]; }

	inline	char*	SymbolName()		{ return ZSYMAM; }
	inline	char*	Laboratory()		{ return ALAB; }
	inline	char*	EvaluationDate()	{ return EDATE;}
	inline	char*	Author()		{ return AUTH; }
	inline	char*	Reference()		{ return REF; }
	inline	char*	DistributionDate()	{ return DDATE; }
	inline	char*	LastRevisionDate()	{ return RDATE; }
	inline	char*	MasterEntryDate()	{ return ENDATE; }
	inline	TString	GetInfo()		{ return INFO; }

	ClassDef(NdbMTDir,1)
}; // NdbMTDir

#endif
