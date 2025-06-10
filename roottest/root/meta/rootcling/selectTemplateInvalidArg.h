#include <TObject.h>

template <class T>
class RtObj2 : public T
 {
  public:
    RtObj2() {}
    RtObj2( const T & val ) : T(val) {}
    ClassDefT(RtObj2,1)
 } ;

ClassDefT2(RtObj2,T)

ClassImpT(RtObj2,T)
