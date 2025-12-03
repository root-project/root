#include <Rtypes.h>
#include <TBuffer.h>

class Foo2 {
public:
  Foo2() {}

private:
  int fData = 3;
  ClassDefNV(Foo2, 1);
};

TBuffer &operator<<(TBuffer &buffer, const Foo2 * /*foo*/) { return buffer; }
TBuffer &operator>>(TBuffer &buffer, Foo2 *&) { return buffer; }
