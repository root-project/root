/* test dict for e.g. streamer generated for class in nested namespaces */

#include "TObject.h"

namespace ns1 {
   namespace ns2 {
      namespace ns3 {
         class SomeClass: public TObject {
         public:
            SomeClass() {}
            virtual ~SomeClass() {}
            ClassDef(SomeClass,1)
         };
      }
   }
}

#ifdef __MAKECINT__
#pragma link C++ class ns1::ns2::ns3::SomeClass+;
#endif
