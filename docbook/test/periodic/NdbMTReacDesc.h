#ifndef NDB_REACDESC_H
#define NDB_REACDESC_H

#include <TArrayI.h>

#include "NdbDefs.h"
#include "NdbMF.h"
#include "NdbMT.h"
#include "NdbParticleList.h"
#include "NdbMaterial.h"

/* ========= NdbMTReactionXS ============ */
// This class provides the descriptions and comments of all the reactions
// that are available in MF=3
class NdbMTReacDesc : public TObject
{
protected:
	TArrayI	mt;		// MT number
	char	**shrt;		//!	Short description (n,f)
	char	**desc;		//!	Long description
	char	**comment;	//!	Comments on reaction	

public:
	NdbMTReacDesc() {
		shrt = NULL;
		desc = NULL;
		comment = NULL;
	}
	NdbMTReacDesc(const char *filename);
	~NdbMTReacDesc();

		void	Init(const char *filename);

	// --- Access functions ---
	inline	Int_t	GetSize()	const	{ return mt.GetSize(); }	
	inline	Int_t	MT(Int_t i)		{ return mt[i]; }

		Int_t	FindMT(Int_t MT);
		char*	GetShort(Int_t MT);
		char*	GetDescription(Int_t MT);
		char*	GetComment(Int_t MT);

	ClassDef(NdbMTReacDesc,1)

}; // NdbMTReacDesc

#endif
