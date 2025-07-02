#include <TNamed.h>

template <class T>
class RtObj : public T
 {
  public:
    RtObj() {}
    RtObj( const T & val ) : T(val) {}
    ClassDefOverride(RtObj,1)
 } ;
