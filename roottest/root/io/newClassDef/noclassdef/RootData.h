#ifndef _ROOT_SIMPLE_RAW_DATA_H
#define _ROOT_SIMPLE_RAW_DATA_H

#include "TClonesArray.h"
#include "TObject.h"

class RootData : public TObject {

public :
	RootData ()  { }
	RootData(char *name, UInt_t nD) : data(name, nD) { }
	~RootData () override { }

	TClonesArray data;
protected:
	ClassDefOverride(RootData,1)
};

#endif /* !defined(_ROOT_SIMPLE_RAW_DATA_H) */

