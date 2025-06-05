#include <TNamed.h>

template <class T>
class RtObj : public T
 {
  public:
    RtObj() {}
    RtObj( const T & val ) : T(val) {}
    ClassDefT(RtObj,1)
 } ;

ClassDefT2(RtObj,T)

ClassImpT(RtObj,T)

