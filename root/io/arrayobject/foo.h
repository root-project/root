#include "TROOT.h"
#include "TClass.h"

class foobj : public TObject {
public:
  foobj() { i = 0; f = 0; }
  foobj(Int_t I) {i = I; f = 2*i; }
  ~foobj() {}

  Int_t i;
  Float_t f;
  
  ClassDef(foobj,1)
};

class foo {
public:
  foo() { i = 0; f = 0; }
  foo(Int_t I) {i = I; f = 2*i; }
  ~foo() {}

  Int_t i;
  Float_t f;
  
  ClassDef(foo,1)
};

//_______________________________________________________________________
inline TBuffer &operator>>(TBuffer &buf, foo *&obj)
{
   // Read foo object from buffer. Declared in ClassDef.
  
   if (!obj) obj = new foo;
   obj->IsA()->ReadBuffer(buf,(void*)obj);
   return buf;
}

inline TBuffer &operator<<(TBuffer &buf, foo *&obj)
{
   // Read foo object from buffer. Declared in ClassDef.
  
  //   if (!obj) obj = new foo;
   obj->IsA()->WriteBuffer(buf,(void*)obj);
   return buf;
}
