#include "TObject.h"
#include "TClonesArray.h"
#include "TRefArray.h"

#ifndef Test_hh
#define Test_hh

class ObjA: public TObject
{
  public:

    UInt_t    fObjAVal;
    void Clear(Option_t* = "") { fObjAVal = 0; }

  ClassDef(ObjA, 1);
};

class Top: public TObject
{
  public:
    Top() { fObjAArray = new TClonesArray(ObjA::Class(), 10); } 
    ~Top() { delete fObjAArray; } 
    void Clear(Option_t* = "")
      { fObjAArray->Clear("C"); fObjAs.Clear(); }

    TClonesArray*    fObjAArray; //->
    TRefArray fObjAs;

  ClassDef(Top, 1);
};
#endif /* Test_hh */
