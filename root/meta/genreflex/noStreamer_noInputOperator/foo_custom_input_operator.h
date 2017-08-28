#include <Rtypes.h>
#include <TBuffer.h>

class Foo {
public:
  Foo() {}

private:
  int fData = 3;
  ClassDefNV(Foo, 1);
};

TBuffer &operator<<(TBuffer &buffer, const Foo *foo) { return buffer; }
TBuffer &operator>>(TBuffer &buffer, Foo *&) { return buffer; }
