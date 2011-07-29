#include "TObject.h"
#include "TClonesArray.h"
#include "TRef.h"

#ifndef execRefClonesArrayTest_h
#define execRefClonesArrayTest_h

class ObjA: public TObject
{
  public:

    UInt_t    fObjAVal;
    TRef      fObjB;

  ClassDef(ObjA, 1);
};

class ObjB: public TObject
{
  public:

    UInt_t fObjBVal;

  ClassDef(ObjB, 1);
};


class Top: public TObject
{
  public:
    Top() { fObjAArray = new TClonesArray(ObjA::Class(), 10); 
            fObjBArray = new TClonesArray(ObjB::Class(), 10); } 
    ~Top() { delete fObjAArray; delete fObjBArray; } 

    TClonesArray*    fObjAArray; //->
    TClonesArray*    fObjBArray; //->
    TRef             fLastB;

  ClassDef(Top, 1);
};
#endif /* execRefClonesArrayTest_h */
