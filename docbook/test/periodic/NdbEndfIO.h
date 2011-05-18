#ifndef __ENDFIO_H
#define __ENDFIO_H

#include <stdio.h>
#include <string.h>

#include <TObject.h>

#include "NdbDefs.h"

// I/O Options
enum {
	TENDF_READ=1,
	TENDF_WRITE=2
};

#define	TENDF_NEXT_NUMBER	-1
#define TENDF_MAXREC		100 

// End records
enum {
	SEND=1,
	FEND=2,
	MEND=3,
	TEND=4
};

class NdbEndfIO : public TObject
{
  // ================== NdbEndfIO =================== 
  // This class is handling all the input/output 
  // to an ENDF-B/VI file
  //
protected:
  	FILE	*f;			//! Working file
	Int_t	mode;			// Open mode
	Int_t	iMAT;			// Material number
	Int_t	iMF;			// Current file type
	Int_t	iMT;			// Current section
	Int_t	lineNum;		// Keep track of current line

	Long_t	matStart;		// Pointer of the current
					// material starting pos
	Long_t	mfStart;		// Current MF starting pos
	Long_t	mtStart;		// Current MT starting pos

	Int_t	lineLen;		// Length of line
  	char	lineTxt[TENDF_MAXREC];	// Complete line
	Int_t	lastNumPos;		// for readReal and readInt;
  static	char	_str[TENDF_MAXREC];	// Used for substrings

public:
	NdbEndfIO()	{ f=NULL; }
	virtual ~NdbEndfIO()
		{ if (f) fclose(f); }

	NdbEndfIO( char *filename, Int_t mode);

  	NdbEndfIO( NdbEndfIO *endf ) {
	  	f	= endf->f;		
	  	mode	= endf->mode;
	 	iMAT	= endf->iMAT;
	 	iMF	= endf->iMF;
	 	iMF	= endf->iMT;
	 	matStart= endf->matStart;
	 	mfStart	= endf->mfStart;
	 	mtStart	= endf->mtStart;
	  	lineNum	= endf->lineNum;
	 	lineLen	= endf->lineLen;
	  	strcpy(lineTxt, endf->lineTxt);
	}


	// Access functions
	inline Int_t	LineNumber()	const { return lineNum; }
	inline char 	*Line()		{ return lineTxt; }
	inline Int_t	MAT()		const { return iMAT; }
	inline Int_t	MF()		const { return iMF; }
	inline Int_t	MT()		const { return iMT; }

	// Is the file open?
	inline	Bool_t	IsOpen()	{ return (f!=NULL); }

	// Have we reached End of file?
	inline	Bool_t	Eof()		{ return feof(f); }

	// End of Section
	inline	Bool_t	EOMT()		{ return (iMT==0); }

	// End of MF
	inline	Bool_t	EOMF()		{ return (iMF==0); }

	// End of MATerial
	inline	Bool_t	EOMAT()		{ return (iMAT==0); }

	// Find Material
	Bool_t		FindMAT( Int_t mat, Bool_t rewind=FALSE );

	// Find File inside material
	Bool_t		FindMATMF( Int_t mat, Int_t mf, Bool_t rewind=FALSE );

	// Find Section
	Bool_t		FindMATMFMT( Int_t mat, Int_t mf, Int_t mt,
					Bool_t rewind=FALSE );

	// Find the specific MF:MT file:section, for current MAT
	Bool_t		FindMFMT(Int_t mf, Int_t mt);

	// Rewind to beggining of current material
	Bool_t		RewindMAT()	{ return fseek(f, matStart, SEEK_SET); }

	// Rewind to beggining of current file (MF)
	Bool_t		RewindMF()	{ return fseek(f, mfStart, SEEK_SET); }

	// Rewind to beggining of current section (MT)
	Bool_t		RewindMT()	{ return fseek(f, mtStart, SEEK_SET); }
		
	// Read entire line
	Bool_t		ReadLine();

	// Reads one by one the real numbers from the current line
	Int_t		ReadInt(Bool_t *error, Int_t pos=TENDF_NEXT_NUMBER);

	Float_t		ReadReal(Bool_t *error, Int_t pos=TENDF_NEXT_NUMBER);

	// Return a substring of current line in a static variable
	char*		Substr(Int_t start, Int_t length);

protected:
	Bool_t		NextNumber(Int_t pos=TENDF_NEXT_NUMBER);
	Int_t		SubReadInt(Int_t start, Int_t length);
	Float_t		SubReadReal(Int_t start, Int_t length);


	ClassDef(NdbEndfIO,1)

}; // NdbEndfIO

#endif
