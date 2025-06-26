#ifndef RefArrayCompress_hh
#define RefArrayCompress_hh

#include "TObject.h"
#include "TClonesArray.h"
#include "TRefArray.h"


class ObjA: public TObject
{
  public:

    UInt_t    fObjAVal;
    void Clear(Option_t* = "") override { fObjAVal = 0; }

  ClassDefOverride(ObjA, 1);
};

class Top: public TObject
{
  public:
    Top() { fObjAArray = new TClonesArray(ObjA::Class(), 10); }
    ~Top() override { delete fObjAArray; }
    void Clear(Option_t* = "") override
      { fObjAArray->Clear("C"); fObjAs.Clear(); }

    TClonesArray*    fObjAArray; //->
    TRefArray fObjAs;

  ClassDefOverride(Top, 1);
};

#endif /* RefArrayCompress_hh */
