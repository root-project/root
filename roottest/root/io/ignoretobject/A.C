// A.C
//
// Class for testing that TTree:Bronch()
// correctly handles TClass::IgnoreTObjectStreamer().
//

#include "TClass.h"
#include "TObject.h"

class A : public TObject {
private:
   double fX;
   double fY;
   double fZ;
public:
   A();
   virtual ~A();
   double get() { return fX*fY*fZ; }
   ClassDef(A, 2);
};

#ifdef __MAKECINT__
#pragma link C++ class A+;
#endif

ClassImp(A);

A::A()
   : fX(0.0),fY(0.0),fZ(0.0)
{
   // We are going to test TTree::Bronch()
   // to see if it handles this correctly. 
   this->Class()->IgnoreTObjectStreamer();
}

A::~A()
{
}

