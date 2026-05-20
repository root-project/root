#include <TObject.h>

template <class T>
class RtObj2 : public T
{
  public:
    RtObj2() {}
    RtObj2( const T & val ) : T(val) {}
    ClassDef(RtObj2,1)
};