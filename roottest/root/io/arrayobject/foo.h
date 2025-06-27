#ifndef FOO_H
#define FOO_H

#include "TROOT.h"
#include "TClass.h"

class foobj : public TObject {
public:
  foobj() { i = 0; f = 0; }
  foobj(Int_t I) {i = I; f = 2*i; }
  virtual ~foobj() {}

  Int_t i;
  Float_t f;
  
  ClassDefOverride(foobj,1)
};

class foo {
public:
  foo() { i = 0; f = 0; }
  foo(Int_t I) {i = I; f = 2*i; }
  virtual ~foo() {}

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

#endif
