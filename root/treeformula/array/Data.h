

#ifndef DATA_H
#define DATA_H

#include <iostream>
#include "TObject.h"


class SubChild : public TObject {
public:
   Float_t efg[5];
   ClassDefOverride(SubChild,1);
};

class NSChild : public TObject {
 public:

   int orient;
   SubChild subs[2];
	Float_t adc[49];            // adc count

	NSChild() {}

	ClassDef (NSChild,1);

};

class Data : public TObject {

 public: 

	NSChild  ns[3];    // 0: East North, 1: East South, 2: West South

	Data() {}

	ClassDef (Data,1);

};


#endif
