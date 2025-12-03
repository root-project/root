#ifndef TestClass_hh
#define TestClass_hh

#include "TObject.h"
#include "TRef.h"
#include <string>
#include <iostream>

class TestClass : public TObject {

  public:

    TestClass( const std::string& name = "");
    TObject* GetRef() { std::cout << "Calling GetRef() from class: " << fName << std::endl;
                        return (fReference.GetObject()); }
    void SetRef(TObject* obj) { fReference = (TObject*)obj; }
    Int_t TestRefBits(UInt_t f) { return fReference.TestBits(f); }
    static void CallOnDemand();

  protected:
    TRef fReference;
    std::string fName;

  ClassDefOverride(TestClass, 1)

};

#endif
