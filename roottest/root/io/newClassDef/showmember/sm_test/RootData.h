#ifndef _ROOT_SIMPLE_RAW_DATA_H
#define _ROOT_SIMPLE_RAW_DATA_H


#include "TClonesArray.h"
#include "TObject.h"


class RootData : public TObject {

public :
	RootData ()  { }
	RootData(char *name, UInt_t nD) : data(name, nD) { }
	virtual ~RootData () {}


	TClonesArray data;
protected:
	ClassDef(RootData,1)
};

#endif /* !defined(_ROOT_SIMPLE_RAW_DATA_H) */




