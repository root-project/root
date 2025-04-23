#include "TObject.h"

class A {
public:
  TBuffer& operator>>(TBuffer&);
};

void g() {
  TBuffer b;
  A *a;
  b >> a;
  b >> *a;
  *a >> b;
}
