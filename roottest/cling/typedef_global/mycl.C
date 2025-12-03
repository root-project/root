#include <vector>

#include "helper.h"

template <class TYPE> class MyClass {
public:
#if defined(BAD)
   // When the '::' are put makecint does not interpreter the typedef 
   // in the vector definition!
    typedef ::ConstLink< TYPE >        value_type ;
#else
    typedef   ConstLink< TYPE >        value_type ;
#endif
    typedef  std::vector< value_type >  collection_type ;

};

MyClass<Toy> m;

